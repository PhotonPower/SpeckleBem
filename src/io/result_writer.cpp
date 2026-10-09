/// @file result_writer.cpp
/// NumPy .npy directory writer (NPY format version 1.0, numpy.lib.format) and the HDF5 stub.
///
/// Values are serialised byte by byte in little-endian order from their bit patterns, so the
/// files are identical on little- and big-endian hosts. Arrays are written in C order through
/// a fixed-size buffer (no full row-major copy of large matrices).
#include "specklebem/io/result_writer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace specklebem::io {

namespace {

namespace fs = std::filesystem;

static_assert(std::numeric_limits<Real>::is_iec559 && sizeof(Real) == 8,
              "the .npy writer needs IEEE-754 binary64 doubles");
static_assert(sizeof(Index) == 8, "the .npy writer stores indices as int64");

/// True if the stem (the part before the first '.') of a component is a Windows device name:
/// CON, PRN, AUX, NUL, COM1-COM9 or LPT1-LPT9, case-insensitive.
bool is_device_name(std::string_view comp) {
    const std::string_view stem = comp.substr(0, comp.find('.'));
    std::string up(stem);
    for (char& c : up) {
        if (c >= 'a' && c <= 'z')
            c = static_cast<char>(c - 'a' + 'A');
    }
    if (up == "CON" || up == "PRN" || up == "AUX" || up == "NUL")
        return true;
    return up.size() == 4 && (up.starts_with("COM") || up.starts_with("LPT")) && up[3] >= '1' &&
           up[3] <= '9';
}

/// Validates a name: '/'-separated components of [A-Za-z0-9_.-], no empty, ".", "..",
/// dot-terminated or Windows device-name components.
void check_name(const std::string& name, std::string_view what) {
    const auto fail = [&](const char* why) {
        throw std::invalid_argument("ResultWriter: invalid " + std::string(what) + " '" + name +
                                    "': " + why);
    };
    if (name.empty())
        fail("empty");
    std::size_t start = 0;
    while (true) {
        const std::size_t end = std::min(name.find('/', start), name.size());
        const std::string_view comp(name.data() + start, end - start);
        if (comp.empty())
            fail("empty component (leading, trailing or repeated '/')");
        if (comp == "." || comp == "..")
            fail("'.' and '..' components are not allowed");
        if (comp.back() == '.')
            fail("components must not end in '.'");
        for (const char c : comp) {
            const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                            (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
            if (!ok)
                fail("allowed characters are [A-Za-z0-9_.-] and '/' between groups");
        }
        if (is_device_name(comp))
            fail("Windows device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9) are not allowed");
        if (end == name.size())
            break;
        start = end + 1;
    }
}

void create_dirs(const fs::path& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir)) {
        throw std::runtime_error("ResultWriter: cannot create directory '" + dir.string() + "'" +
                                 (ec ? ": " + ec.message() : std::string()));
    }
}

/// Output file written to a temporary file next to the target ("<target>~tmp"; '~' cannot
/// occur in a valid name) and renamed over the target by commit(), so the target is either the
/// complete old or the complete new file. Without commit() the temporary file is removed.
class AtomicFile {
public:
    explicit AtomicFile(fs::path path) : path_(std::move(path)), tmp_(path_) {
        tmp_ += "~tmp";
        std::error_code ec;
        if (fs::is_directory(path_, ec))
            fail("is a directory");
        out_.open(tmp_, std::ios::binary | std::ios::trunc);
        if (!out_)
            fail("cannot open temporary file '" + tmp_.string() + "' for writing");
    }
    AtomicFile(const AtomicFile&) = delete;
    AtomicFile& operator=(const AtomicFile&) = delete;
    ~AtomicFile() {
        if (committed_)
            return;
        if (out_.is_open())
            out_.close();
        std::error_code ec;
        fs::remove(tmp_, ec);
    }

    void write(std::string_view bytes) {
        out_.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!out_)
            fail("write error");
    }

    /// Closes the temporary file and renames it over the target (replacing an existing file).
    void commit() {
        out_.close();
        if (!out_)
            fail("write error");
        std::error_code ec;
        fs::rename(tmp_, path_, ec);
        if (ec)
            fail("cannot rename '" + tmp_.string() + "' over it: " + ec.message());
        committed_ = true;
    }

private:
    [[noreturn]] void fail(const std::string& why) const {
        throw std::runtime_error("ResultWriter: '" + path_.string() + "': " + why);
    }

    fs::path path_;
    fs::path tmp_;
    std::ofstream out_;
    bool committed_ = false;
};

void put_le64(std::string& buf, std::uint64_t bits) {
    for (int b = 0; b < 8; ++b) buf.push_back(static_cast<char>((bits >> (8 * b)) & 0xffU));
}
void put_value(std::string& buf, Real x) {
    put_le64(buf, std::bit_cast<std::uint64_t>(x));
}
void put_value(std::string& buf, Complex z) {
    put_value(buf, z.real());
    put_value(buf, z.imag());
}
void put_value(std::string& buf, Index i) {
    put_le64(buf, std::bit_cast<std::uint64_t>(i));
}

/// NPY 1.0 preamble: magic, version, uint16 LE header length, dict padded with spaces and
/// terminated by '\n' so that the preamble length is a multiple of 64.
std::string npy_header(std::string_view descr, Index rows, Index cols, bool is_vector) {
    std::string dict = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': (";
    dict += std::to_string(rows);
    dict += is_vector ? std::string(",") : ", " + std::to_string(cols);
    dict += "), }";
    constexpr std::size_t kPrefix = 10;  // magic (6) + version (2) + header length (2)
    const std::size_t total = (kPrefix + dict.size() + 1 + 63) / 64 * 64;
    dict.append(total - kPrefix - dict.size() - 1, ' ');
    dict.push_back('\n');
    const std::size_t len = dict.size();
    std::string out("\x93NUMPY\x01\x00", 8);
    out.push_back(static_cast<char>(len & 0xffU));
    out.push_back(static_cast<char>((len >> 8) & 0xffU));
    return out + dict;
}

/// Writes a (rows x cols) array in C order; at(i, j) returns the element.
template <class At>
void write_npy(const fs::path& path, std::string_view descr, Index rows, Index cols, bool is_vector,
               const At& at) {
    AtomicFile out(path);
    std::string buf = npy_header(descr, rows, cols, is_vector);
    constexpr std::size_t kChunk = std::size_t{1} << 16;
    for (Index i = 0; i < rows; ++i) {
        for (Index j = 0; j < cols; ++j) put_value(buf, at(i, j));
        if (buf.size() >= kChunk) {
            out.write(buf);
            buf.clear();
        }
    }
    out.write(buf);
    out.commit();
}

/// Throws std::invalid_argument unless s is well-formed UTF-8 (no overlongs, surrogates or
/// code points above U+10FFFF).
void check_utf8(const std::string& s, const std::string& name) {
    const auto byte = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
    std::size_t k = 0;
    while (k < s.size()) {
        const unsigned char c = byte(k);
        std::size_t n = 0;
        std::uint32_t cp = 0;
        if (c < 0x80U) {
            ++k;
            continue;
        } else if ((c & 0xe0U) == 0xc0U) {
            n = 1;
            cp = c & 0x1fU;
        } else if ((c & 0xf0U) == 0xe0U) {
            n = 2;
            cp = c & 0x0fU;
        } else if ((c & 0xf8U) == 0xf0U) {
            n = 3;
            cp = c & 0x07U;
        } else {
            n = 99;
        }
        bool ok = n <= 3 && k + n < s.size();  // continuation bytes at k+1 .. k+n
        for (std::size_t m = 1; ok && m <= n; ++m) {
            ok = (byte(k + m) & 0xc0U) == 0x80U;
            if (ok)
                cp = (cp << 6) | (byte(k + m) & 0x3fU);
        }
        constexpr std::array<std::uint32_t, 4> kMin = {0, 0x80U, 0x800U, 0x10000U};
        ok = ok && cp >= kMin[n] && cp <= 0x10ffffU && (cp < 0xd800U || cp > 0xdfffU);
        if (!ok) {
            throw std::invalid_argument("ResultWriter: attribute '" + name +
                                        "' is not valid UTF-8 (byte " + std::to_string(k) + ")");
        }
        k += n + 1;
    }
}

std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (const char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        if (ch == '"' || ch == '\\') {
            out.push_back('\\');
            out.push_back(ch);
        } else if (ch == '\n') {
            out += "\\n";
        } else if (ch == '\t') {
            out += "\\t";
        } else if (c < 0x20U) {  // other control characters as \u00XX
            constexpr char kHex[] = "0123456789abcdef";
            out += "\\u00";
            out.push_back(kHex[c >> 4]);
            out.push_back(kHex[c & 0xfU]);
        } else {
            out.push_back(ch);  // UTF-8 passes through unescaped
        }
    }
    return out + '"';
}

std::string json_number(Real x) {
    std::array<char, 32> buf{};
    const auto res = std::to_chars(buf.data(), buf.data() + buf.size(), x);
    std::string s(buf.data(), res.ptr);
    if (s.find_first_of(".e") == std::string::npos)
        s += ".0";
    return s;
}

class NpyDirectoryWriter final : public ResultWriter {
public:
    explicit NpyDirectoryWriter(fs::path root) : root_(std::move(root)) { create_dirs(root_); }

    void write_mesh(const std::string& group, const geometry::TriangleMesh& m) override {
        check_name(group, "group");
        const fs::path dir = root_ / fs::path(group);
        create_dirs(dir);
        const Vertices& v = m.vertices();
        const Triangles& t = m.triangles();
        write_npy(dir / "vertices.npy", "<f8", v.rows(), 3, false,
                  [&](Index i, Index j) { return v(i, j); });
        write_npy(dir / "triangles.npy", "<i8", t.rows(), 3, false,
                  [&](Index i, Index j) { return t(i, j); });
    }
    void write_vector(const std::string& name, const VectorXc& v) override {
        write_npy(array_path(name), "<c16", v.size(), 1, true,
                  [&](Index i, Index) { return v(i); });
    }
    void write_matrix(const std::string& name, const MatrixXr& m) override {
        write_npy(array_path(name), "<f8", m.rows(), m.cols(), false,
                  [&](Index i, Index j) { return m(i, j); });
    }
    void write_matrix(const std::string& name, const MatrixXc& m) override {
        write_npy(array_path(name), "<c16", m.rows(), m.cols(), false,
                  [&](Index i, Index j) { return m(i, j); });
    }
    void write_attribute(const std::string& name, const std::string& value) override {
        check_name(name, "attribute name");
        check_utf8(value, name);
        set_attribute(name, json_string(value));
    }
    void write_attribute(const std::string& name, Real value) override {
        check_name(name, "attribute name");
        if (!std::isfinite(value)) {
            throw std::invalid_argument("ResultWriter: attribute '" + name +
                                        "' is not finite (no JSON representation)");
        }
        set_attribute(name, json_number(value));
    }

private:
    /// Validated <root>/<name>.npy; creates the group directories.
    fs::path array_path(const std::string& name) const {
        check_name(name, "name");
        const fs::path path = root_ / fs::path(name + ".npy");
        create_dirs(path.parent_path());
        return path;
    }

    /// Stores (or replaces) the attribute and rewrites attributes.json completely.
    void set_attribute(const std::string& name, std::string json_value) {
        bool found = false;
        for (auto& [key, val] : attributes_) {
            if (key == name) {
                val = std::move(json_value);
                found = true;
                break;
            }
        }
        if (!found)
            attributes_.emplace_back(name, std::move(json_value));
        std::string text = "{";
        for (std::size_t k = 0; k < attributes_.size(); ++k) {
            text += k == 0 ? "\n  " : ",\n  ";
            text += json_string(attributes_[k].first) + ": " + attributes_[k].second;
        }
        text += "\n}\n";
        const fs::path path = root_ / "attributes.json";
        AtomicFile out(path);
        out.write(text);
        out.commit();
    }

    fs::path root_;
    std::vector<std::pair<std::string, std::string>> attributes_;  ///< name, JSON value
};

}  // namespace

std::unique_ptr<ResultWriter> open_hdf5(const std::string& path) {
#ifdef SPECKLEBEM_HAVE_HDF5
    throw std::runtime_error("open_hdf5('" + path +
                             "'): the HDF5 writer is not implemented yet; use open_npy_directory");
#else
    throw std::runtime_error("open_hdf5('" + path +
                             "'): SpeckleBem was built without HDF5 support "
                             "(SPECKLEBEM_ENABLE_HDF5=OFF); use open_npy_directory");
#endif
}

std::unique_ptr<ResultWriter> open_npy_directory(const std::string& dir) {
    if (dir.empty())
        throw std::invalid_argument("open_npy_directory: empty directory path");
    return std::make_unique<NpyDirectoryWriter>(fs::path(dir));
}

}  // namespace specklebem::io
