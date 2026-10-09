/// @file mesh_io.cpp
/// STL (binary / ASCII), Wavefront OBJ and Gmsh MSH 4.1 ASCII import / export.
///
/// Files are read into memory in one piece and tokenised by hand (std::from_chars for
/// numbers, no regex, no per-token stream extraction), so that meshes with ~10^6
/// triangles load in about a second. Writers format into a buffer that is flushed in
/// chunks; doubles are written in their shortest round-trip form (std::to_chars).
#include "specklebem/geometry/mesh_io.hpp"

#include "specklebem/core/logging.hpp"
#include "specklebem/core/path.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace specklebem::geometry::io {

namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "binary STL needs IEEE-754 binary32 floats");

// ---------------------------------------------------------------------------
// Errors and helpers
// ---------------------------------------------------------------------------

/// Throws std::runtime_error "fn: 'path'[, line L]: msg" (line <= 0: no line number).
[[noreturn]] void fail(std::string_view fn, const std::string& path, Index line,
                       const std::string& msg) {
    std::string s(fn);
    s += ": '";
    s += path;
    s += '\'';
    if (line > 0) {
        s += ", line ";
        s += std::to_string(line);
    }
    s += ": ";
    s += msg;
    throw std::runtime_error(s);
}

/// Token quoted for an error message: truncated, non-printable bytes replaced.
std::string quote(std::string_view tok) {
    constexpr std::size_t kMax = 40;
    std::string s = "'";
    for (std::size_t i = 0; i < tok.size() && i < kMax; ++i) {
        const auto c = static_cast<unsigned char>(tok[i]);
        s += (c >= 0x20 && c < 0x7f) ? tok[i] : '?';
    }
    if (tok.size() > kMax)
        s += "...";
    s += '\'';
    return s;
}

char to_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (to_lower(a[i]) != to_lower(b[i]))
            return false;
    }
    return true;
}

/// Lower-case extension of a UTF-8 path. @throws std::invalid_argument for invalid UTF-8.
std::string lower_extension(const std::string& path) {
    std::string ext = core::path_to_utf8(core::path_from_utf8(path).extension());
    std::transform(ext.begin(), ext.end(), ext.begin(), to_lower);
    return ext;
}

/// Whole file as a byte string. @throws std::runtime_error naming the path,
/// std::invalid_argument if the path is not valid UTF-8.
std::string read_file(std::string_view fn, const std::string& path) {
    const std::filesystem::path p = core::path_from_utf8(path);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(p, ec)) {
        fail(fn, path, 0,
             ec ? "cannot open file (" + ec.message() + ")"
                : std::string("cannot open file (it does not exist or is not a regular file)"));
    }
    const std::uintmax_t size = std::filesystem::file_size(p, ec);
    if (ec)
        fail(fn, path, 0, "cannot determine file size (" + ec.message() + ")");
    std::ifstream in(p, std::ios::binary);
    if (!in)
        fail(fn, path, 0, "cannot open file for reading");
    std::string data(static_cast<std::size_t>(size), '\0');
    if (size > 0 && !in.read(data.data(), static_cast<std::streamsize>(size)))
        fail(fn, path, 0, "read error");
    return data;
}

/// Constructs the mesh; constructor rejections become runtime_errors naming the file.
TriangleMesh make_mesh(std::string_view fn, const std::string& path, Vertices vertices,
                       Triangles triangles) {
    if (triangles.rows() == 0)
        fail(fn, path, 0, "no triangles found");
    try {
        return TriangleMesh(std::move(vertices), std::move(triangles));
    } catch (const std::invalid_argument& e) {
        fail(fn, path, 0, std::string("invalid mesh: ") + e.what());
    }
}

void require_triangles(const TriangleMesh& m, std::string_view fn, const std::string& path) {
    if (m.num_triangles() == 0) {
        throw std::invalid_argument(std::string(fn) + ": '" + path +
                                    "': mesh has no triangles, nothing to write");
    }
}

// ---------------------------------------------------------------------------
// Tokenizer: whitespace-separated tokens with line tracking.
// ---------------------------------------------------------------------------

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

/// True if text[pos] ends a line: '\n', or a '\r' not followed by '\n' (CR-only files).
/// In CRLF files the '\r' is ordinary whitespace and the '\n' ends the line.
bool is_eol(std::string_view text, std::size_t pos) {
    return text[pos] == '\n' ||
           (text[pos] == '\r' && (pos + 1 == text.size() || text[pos + 1] != '\n'));
}

/// Length of a leading UTF-8 byte-order mark (0 or 3).
std::size_t bom_length(std::string_view text) {
    return text.substr(0, 3) == "\xEF\xBB\xBF" ? 3 : 0;
}

class Tokenizer {
public:
    Tokenizer(std::string_view text, std::string_view fn, const std::string& path)
        : text_(text), fn_(fn), path_(path), pos_(bom_length(text)) {}

    /// Next token (across lines). False at end of input.
    bool next(std::string_view& tok) {
        while (pos_ < text_.size() && is_space(text_[pos_])) {
            if (is_eol(text_, pos_))
                ++line_;
            ++pos_;
        }
        return read_token(tok);
    }

    /// Next token on the current line. False at end of line / input (newline not consumed).
    bool next_on_line(std::string_view& tok) {
        while (pos_ < text_.size() && is_space(text_[pos_]) && !is_eol(text_, pos_)) ++pos_;
        if (pos_ < text_.size() && is_eol(text_, pos_))
            return false;
        return read_token(tok);
    }

    /// Skips the rest of the current line including its newline.
    void skip_line() {
        while (pos_ < text_.size() && !is_eol(text_, pos_)) ++pos_;
        if (pos_ < text_.size()) {
            ++pos_;
            ++line_;
        }
    }

    std::string_view expect(std::string_view what) {
        std::string_view tok;
        if (!next(tok)) {
            tok_line_ = line_;
            error("unexpected end of file, expected " + std::string(what));
        }
        return tok;
    }

    void expect_keyword(std::string_view kw, bool case_insensitive = false) {
        const std::string_view tok = expect("'" + std::string(kw) + "'");
        if (case_insensitive ? !iequals(tok, kw) : tok != kw)
            error("expected '" + std::string(kw) + "', found " + quote(tok));
    }

    Real expect_real(std::string_view what) { return to_real(expect(what), what); }
    Index expect_index(std::string_view what) { return to_index(expect(what), what); }

    /// Finite double; a leading '+' is accepted.
    Real to_real(std::string_view tok, std::string_view what) const {
        std::string_view s = tok;
        if (!s.empty() && s.front() == '+')
            s.remove_prefix(1);
        Real value = 0.0;
        const char* end = s.data() + s.size();
        const auto [ptr, ec] = std::from_chars(s.data(), end, value);
        if (s.empty() || s.front() == '+' || (tok.front() == '+' && s.front() == '-') ||
            ec != std::errc() || ptr != end) {
            error("invalid number " + quote(tok) + " (" + std::string(what) + ")");
        }
        if (!std::isfinite(value))
            error("non-finite value " + quote(tok) + " (" + std::string(what) + ")");
        return value;
    }

    Index to_index(std::string_view tok, std::string_view what) const {
        Index value = 0;
        const char* end = tok.data() + tok.size();
        const auto [ptr, ec] = std::from_chars(tok.data(), end, value);
        if (tok.empty() || ec != std::errc() || ptr != end)
            error("invalid integer " + quote(tok) + " (" + std::string(what) + ")");
        return value;
    }

    /// Line number of the last token read.
    [[nodiscard]] Index line() const { return tok_line_; }

    [[noreturn]] void error(const std::string& msg) const { fail(fn_, path_, tok_line_, msg); }

private:
    bool read_token(std::string_view& tok) {
        if (pos_ >= text_.size())
            return false;
        const std::size_t start = pos_;
        while (pos_ < text_.size() && !is_space(text_[pos_])) ++pos_;
        tok = text_.substr(start, pos_ - start);
        tok_line_ = line_;
        return true;
    }

    std::string_view text_;
    std::string_view fn_;
    const std::string& path_;
    std::size_t pos_;
    Index line_ = 1;
    Index tok_line_ = 1;
};

// ---------------------------------------------------------------------------
// Exact vertex welding (STL)
// ---------------------------------------------------------------------------

struct VertexKey {
    Real x, y, z;
    bool operator==(const VertexKey&) const = default;  // -0 == +0
};

struct VertexKeyHash {
    std::size_t operator()(const VertexKey& k) const noexcept {
        const std::hash<Real> h;
        // Map -0 to +0 so that equal keys hash equally.
        std::size_t seed = h(k.x == 0.0 ? 0.0 : k.x);
        for (const Real c : {k.y, k.z}) {
            seed ^= h(c == 0.0 ? 0.0 : c) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

/// Assigns indices to vertices in order of first appearance; exact coordinate match.
class Welder {
public:
    explicit Welder(std::size_t expected_vertices) {
        map_.reserve(expected_vertices);
        coords_.reserve(3 * expected_vertices);
    }

    Index add(Real x, Real y, Real z) {
        const auto [it, inserted] =
            map_.try_emplace(VertexKey{x, y, z}, static_cast<Index>(coords_.size() / 3));
        if (inserted) {
            coords_.push_back(x);
            coords_.push_back(y);
            coords_.push_back(z);
        }
        return it->second;
    }

    [[nodiscard]] Vertices vertices() const {
        const auto n = static_cast<Index>(coords_.size() / 3);
        Vertices v(n, 3);
        if (n > 0)
            std::memcpy(v.data(), coords_.data(), coords_.size() * sizeof(Real));
        return v;
    }

private:
    std::unordered_map<VertexKey, Index, VertexKeyHash> map_;
    std::vector<Real> coords_;
};

Triangles to_triangles(const std::vector<Index>& corners) {
    const auto n = static_cast<Index>(corners.size() / 3);
    Triangles t(n, 3);
    if (n > 0)
        std::memcpy(t.data(), corners.data(), corners.size() * sizeof(Index));
    return t;
}

// ---------------------------------------------------------------------------
// Little-endian binary helpers
// ---------------------------------------------------------------------------

std::uint32_t load_u32(const std::string& data, std::size_t off) {
    std::uint32_t u = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        u |= static_cast<std::uint32_t>(static_cast<unsigned char>(data[off + i])) << (8 * i);
    }
    return u;
}

float load_f32(const std::string& data, std::size_t off) {
    const std::uint32_t u = load_u32(data, off);
    float f = 0.0F;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

// ---------------------------------------------------------------------------
// Buffered output file
// ---------------------------------------------------------------------------

/// Writes to `path + ".tmp"` and renames it to `path` in close(), so that a failed write
/// never leaves a truncated file at the target. The temporary file is removed if close()
/// is not reached or fails. `path` is UTF-8 (std::invalid_argument otherwise); messages name
/// the UTF-8 strings.
class OutputFile {
public:
    OutputFile(std::string_view fn, const std::string& path)
        : fn_(fn),
          path_(path),
          target_(core::path_from_utf8(path)),
          tmp_(core::path_from_utf8(path + ".tmp")) {
        out_.open(tmp_, std::ios::binary | std::ios::trunc);
        if (!out_)
            fail(fn_, path_, 0, "cannot open '" + path_ + ".tmp' for writing");
        buf_.reserve(kFlushSize + 256);
    }
    OutputFile(const OutputFile&) = delete;
    OutputFile& operator=(const OutputFile&) = delete;
    ~OutputFile() {
        if (!committed_)
            discard();
    }

    void put(std::string_view s) {
        buf_.append(s);
        maybe_flush();
    }
    void put(char c) {
        buf_.push_back(c);
        maybe_flush();
    }
    /// Shortest decimal representation that round-trips exactly (<= 17 significant digits).
    void put_real(Real x) {
        std::array<char, 32> b{};
        const auto r = std::to_chars(b.data(), b.data() + b.size(), x);
        buf_.append(b.data(), r.ptr);
        maybe_flush();
    }
    void put_index(Index i) {
        std::array<char, 24> b{};
        const auto r = std::to_chars(b.data(), b.data() + b.size(), i);
        buf_.append(b.data(), r.ptr);
        maybe_flush();
    }
    void put_u16(std::uint16_t u) {
        buf_.push_back(static_cast<char>(u & 0xFFU));
        buf_.push_back(static_cast<char>((u >> 8) & 0xFFU));
        maybe_flush();
    }
    void put_u32(std::uint32_t u) {
        for (unsigned i = 0; i < 4; ++i) buf_.push_back(static_cast<char>((u >> (8 * i)) & 0xFFU));
        maybe_flush();
    }
    void put_f32(Real x) {
        const auto f = static_cast<float>(x);
        std::uint32_t u = 0;
        std::memcpy(&u, &f, sizeof u);
        put_u32(u);
    }

    /// Flushes, closes and moves the temporary file to the target path.
    void close() {
        flush();
        out_.close();
        if (!out_)
            fail(fn_, path_, 0, "write error");
        std::error_code ec;
        std::filesystem::rename(tmp_, target_, ec);
        if (ec)
            fail(fn_, path_, 0, "cannot rename '" + path_ + ".tmp' (" + ec.message() + ")");
        committed_ = true;
    }

private:
    static constexpr std::size_t kFlushSize = std::size_t{1} << 20;

    void maybe_flush() {
        if (buf_.size() >= kFlushSize)
            flush();
    }
    void flush() {
        out_.write(buf_.data(), static_cast<std::streamsize>(buf_.size()));
        buf_.clear();
        if (!out_)
            fail(fn_, path_, 0, "write error");
    }

    void discard() noexcept {
        if (out_.is_open())
            out_.close();
        std::error_code ec;
        std::filesystem::remove(tmp_, ec);
    }

    std::string_view fn_;
    const std::string& path_;  ///< UTF-8, for messages
    std::filesystem::path target_;
    std::filesystem::path tmp_;
    std::ofstream out_;
    std::string buf_;
    bool committed_ = false;
};

// ---------------------------------------------------------------------------
// STL
// ---------------------------------------------------------------------------

constexpr std::size_t kStlHeader = 84;  // 80-byte header + uint32 count
constexpr std::size_t kStlRecord = 50;

TriangleMesh parse_stl_binary(const std::string& data, const std::string& path, std::uint32_t n) {
    constexpr std::string_view fn = "read_stl";
    Welder welder(n / 2 + 3);
    Triangles tris(static_cast<Index>(n), 3);
    for (std::uint32_t t = 0; t < n; ++t) {
        const std::size_t base = kStlHeader + kStlRecord * std::size_t{t} + 12;  // skip normal
        for (std::size_t k = 0; k < 3; ++k) {
            std::array<Real, 3> p{};
            for (std::size_t c = 0; c < 3; ++c) {
                p[c] = static_cast<Real>(load_f32(data, base + 12 * k + 4 * c));
                if (!std::isfinite(p[c])) {
                    fail(fn, path, 0,
                         "binary STL triangle " + std::to_string(t) +
                             " has a non-finite vertex coordinate");
                }
            }
            tris(static_cast<Index>(t), static_cast<Index>(k)) = welder.add(p[0], p[1], p[2]);
        }
    }
    return make_mesh(fn, path, welder.vertices(), std::move(tris));
}

TriangleMesh parse_stl_ascii(std::string_view data, const std::string& path) {
    constexpr std::string_view fn = "read_stl";
    Tokenizer tz(data, fn, path);
    Welder welder(data.size() / 512 + 16);  // ~250 bytes per facet, ~F/2 vertices
    std::vector<Index> corners;
    corners.reserve(data.size() / 80);

    tz.expect_keyword("solid", true);
    tz.skip_line();  // solid name
    std::string_view tok;
    for (;;) {
        if (!tz.next(tok))
            tz.error("unexpected end of file, expected 'facet' or 'endsolid'");
        if (iequals(tok, "facet")) {
            tz.expect_keyword("normal", true);
            for (int c = 0; c < 3; ++c) tz.expect("facet normal component");  // ignored
            tz.expect_keyword("outer", true);
            tz.expect_keyword("loop", true);
            for (int k = 0; k < 3; ++k) {
                tz.expect_keyword("vertex", true);
                const Real x = tz.expect_real("vertex x");
                const Real y = tz.expect_real("vertex y");
                const Real z = tz.expect_real("vertex z");
                corners.push_back(welder.add(x, y, z));
            }
            tz.expect_keyword("endloop", true);
            tz.expect_keyword("endfacet", true);
        } else if (iequals(tok, "endsolid")) {
            tz.skip_line();  // solid name
            if (!tz.next(tok))
                break;
            if (!iequals(tok, "solid"))
                tz.error("expected 'solid' or end of file after 'endsolid', found " + quote(tok));
            tz.skip_line();
        } else {
            tz.error("expected 'facet' or 'endsolid', found " + quote(tok));
        }
    }
    return make_mesh(fn, path, welder.vertices(), to_triangles(corners));
}

bool starts_with_solid(const std::string& data) {
    std::size_t i = bom_length(data);
    while (i < data.size() && is_space(data[i])) ++i;
    return data.size() - i >= 5 && iequals(std::string_view(data).substr(i, 5), "solid");
}

/// Why `data` is not a binary STL (size mismatch for the stored count, or too short).
std::string binary_stl_mismatch(const std::string& data) {
    if (data.size() < kStlHeader)
        return "file shorter than the 84-byte binary header";
    const std::uint64_t n = load_u32(data, 80);
    return "file size " + std::to_string(data.size()) + " bytes != 84 + 50 x " + std::to_string(n) +
           " = " + std::to_string(kStlHeader + kStlRecord * n) +
           " for the stored binary triangle count " + std::to_string(n);
}

void write_stl_binary(const TriangleMesh& m, const std::string& path) {
    constexpr std::string_view fn = "write_stl";
    if (m.num_triangles() > static_cast<Index>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::invalid_argument("write_stl: '" + path +
                                    "': binary STL holds at most 2^32 - 1 triangles");
    }
    OutputFile out(fn, path);
    std::string header = "SpeckleBem binary STL, units: metres";  // must not start with "solid"
    header.resize(80, ' ');
    out.put(header);
    out.put_u32(static_cast<std::uint32_t>(m.num_triangles()));
    const Vertices& v = m.vertices();
    const Triangles& tri = m.triangles();
    for (Index t = 0; t < m.num_triangles(); ++t) {
        const Vec3 n = m.normal(t);
        out.put_f32(n.x());
        out.put_f32(n.y());
        out.put_f32(n.z());
        for (Index k = 0; k < 3; ++k) {
            for (Index c = 0; c < 3; ++c) out.put_f32(v(tri(t, k), c));
        }
        out.put_u16(0);
    }
    out.close();
}

void write_stl_ascii(const TriangleMesh& m, const std::string& path) {
    OutputFile out("write_stl", path);
    out.put("solid specklebem\n");
    const Vertices& v = m.vertices();
    const Triangles& tri = m.triangles();
    for (Index t = 0; t < m.num_triangles(); ++t) {
        const Vec3 n = m.normal(t);
        out.put("  facet normal ");
        out.put_real(n.x());
        out.put(' ');
        out.put_real(n.y());
        out.put(' ');
        out.put_real(n.z());
        out.put("\n    outer loop\n");
        for (Index k = 0; k < 3; ++k) {
            out.put("      vertex ");
            out.put_real(v(tri(t, k), 0));
            out.put(' ');
            out.put_real(v(tri(t, k), 1));
            out.put(' ');
            out.put_real(v(tri(t, k), 2));
            out.put('\n');
        }
        out.put("    endloop\n  endfacet\n");
    }
    out.put("endsolid specklebem\n");
    out.close();
}

// ---------------------------------------------------------------------------
// Gmsh element type names (for error messages)
// ---------------------------------------------------------------------------

std::string gmsh_element_name(Index type) {
    switch (type) {
        case 3:
            return "4-node quadrangle";
        case 4:
            return "4-node tetrahedron";
        case 5:
            return "8-node hexahedron";
        case 6:
            return "6-node prism";
        case 7:
            return "5-node pyramid";
        case 8:
            return "3-node second-order line";
        case 9:
            return "6-node second-order triangle";
        case 10:
            return "9-node second-order quadrangle";
        case 11:
            return "10-node second-order tetrahedron";
        default:
            return "unsupported element";
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

TriangleMesh read_mesh(const std::string& path) {
    const std::string ext = lower_extension(path);
    if (ext == ".stl")
        return read_stl(path);
    if (ext == ".obj")
        return read_obj(path);
    if (ext == ".msh")
        return read_gmsh(path);
    throw std::invalid_argument("read_mesh: '" + path + "': unknown mesh file extension '" + ext +
                                "' (supported: .stl, .obj, .msh)");
}

void write_mesh(const TriangleMesh& m, const std::string& path) {
    const std::string ext = lower_extension(path);
    if (ext == ".stl")
        return write_stl(m, path, StlFormat::binary);
    if (ext == ".obj")
        return write_obj(m, path);
    if (ext == ".msh")
        return write_gmsh(m, path);
    throw std::invalid_argument("write_mesh: '" + path + "': unknown mesh file extension '" + ext +
                                "' (supported: .stl, .obj, .msh)");
}

TriangleMesh read_stl(const std::string& path) {
    const std::string data = read_file("read_stl", path);
    if (data.size() >= kStlHeader) {
        const std::uint32_t n = load_u32(data, 80);
        if (std::uint64_t{data.size()} == kStlHeader + kStlRecord * std::uint64_t{n})
            return parse_stl_binary(data, path, n);
    }
    if (!starts_with_solid(data)) {
        fail("read_stl", path, 0,
             "neither a binary STL (" + binary_stl_mismatch(data) +
                 ") nor an ASCII STL (does not start with 'solid')");
    }
    try {
        return parse_stl_ascii(data, path);
    } catch (const std::runtime_error& e) {
        // A truncated binary STL whose header starts with "solid" ends up here.
        if (data.size() < kStlHeader)
            throw;
        throw std::runtime_error(std::string(e.what()) + " [parsed as ASCII STL because " +
                                 binary_stl_mismatch(data) + "]");
    }
}

TriangleMesh read_obj(const std::string& path) {
    constexpr std::string_view fn = "read_obj";
    const std::string data = read_file(fn, path);
    Tokenizer tz(data, fn, path);
    std::vector<Real> coords;
    std::vector<Index> corners;
    std::vector<Index> triangle_line;  // source line of each triangle (index errors)
    std::vector<Index> polygon;
    Index num_polygons = 0;

    std::string_view tok;
    while (tz.next(tok)) {
        if (tok == "v") {
            for (int c = 0; c < 3; ++c) {
                std::string_view value;
                if (!tz.next_on_line(value))
                    tz.error("vertex statement needs three coordinates");
                coords.push_back(tz.to_real(value, "vertex coordinate"));
            }
        } else if (tok == "f") {
            polygon.clear();
            std::string_view ref;
            bool comment = false;
            while (!comment && tz.next_on_line(ref)) {
                const std::size_t hash = ref.find('#');  // "3#comment": rest of line is comment
                if (hash != std::string_view::npos) {
                    comment = true;
                    ref = ref.substr(0, hash);
                    if (ref.empty())
                        break;
                }
                const std::string_view first = ref.substr(0, ref.find('/'));
                if (first.empty())
                    tz.error("face vertex " + quote(ref) + " has no vertex index");
                const Index idx = tz.to_index(first, "face vertex index");
                if (idx < 0) {
                    tz.error("negative (relative) vertex index " + std::to_string(idx) +
                             " is not supported");
                }
                if (idx == 0)
                    tz.error("vertex index 0 is invalid (OBJ indices are 1-based)");
                polygon.push_back(idx - 1);
            }
            if (polygon.size() < 3)
                tz.error("face statement needs at least three vertices");
            if (polygon.size() > 3)
                ++num_polygons;
            for (std::size_t i = 1; i + 1 < polygon.size(); ++i) {
                corners.push_back(polygon[0]);
                corners.push_back(polygon[i]);
                corners.push_back(polygon[i + 1]);
                triangle_line.push_back(tz.line());
            }
        }
        tz.skip_line();  // remainder of v lines, all other statements and comments
    }

    const auto num_vertices = static_cast<Index>(coords.size() / 3);
    for (std::size_t c = 0; c < corners.size(); ++c) {
        if (corners[c] >= num_vertices) {
            fail(fn, path, triangle_line[c / 3],
                 "face vertex index " + std::to_string(corners[c] + 1) +
                     " out of range (the file has " + std::to_string(num_vertices) + " vertices)");
        }
    }
    if (num_polygons > 0) {
        SBEM_WARN("read_obj: '{}': {} face(s) with more than three vertices fan-triangulated", path,
                  num_polygons);
    }
    Vertices vertices(num_vertices, 3);
    if (num_vertices > 0)
        std::memcpy(vertices.data(), coords.data(), coords.size() * sizeof(Real));
    return make_mesh(fn, path, std::move(vertices), to_triangles(corners));
}

TriangleMesh read_gmsh(const std::string& path) {
    constexpr std::string_view fn = "read_gmsh";
    const std::string data = read_file(fn, path);
    Tokenizer tz(data, fn, path);
    // Upper bound for reservations, robust against absurd counts in corrupt headers.
    const auto max_items = static_cast<Index>(data.size() / 2 + 1);

    bool have_format = false;
    bool have_nodes = false;
    bool have_elements = false;
    std::vector<Real> node_coords;
    std::unordered_map<Index, Index> node_of_tag;
    std::vector<Index> corners;  // node indices

    std::string_view tok;
    while (tz.next(tok)) {
        if (tok == "$MeshFormat") {
            const std::string_view version = tz.expect("format version");
            const std::string_view file_type = tz.expect("file type");
            const std::string_view data_size = tz.expect("data size");
            if (version != "4.1" || file_type != "0" || data_size != "8") {
                tz.error("unsupported MSH format version '" + std::string(version) + " " +
                         std::string(file_type) + " " + std::string(data_size) +
                         "'; only Gmsh MSH 4.1 ASCII ('4.1 0 8') is supported");
            }
            tz.expect_keyword("$EndMeshFormat");
            have_format = true;
        } else if (tok == "$Nodes") {
            if (!have_format)
                tz.error("$Nodes before $MeshFormat");
            if (have_nodes)
                tz.error("more than one $Nodes section");
            const Index num_blocks = tz.expect_index("number of node entity blocks");
            const Index num_nodes = tz.expect_index("number of nodes");
            tz.expect_index("minimum node tag");
            tz.expect_index("maximum node tag");
            if (num_blocks < 0 || num_nodes < 0)
                tz.error("negative node count");
            node_of_tag.reserve(static_cast<std::size_t>(std::min(num_nodes, max_items)));
            node_coords.reserve(3 * static_cast<std::size_t>(std::min(num_nodes, max_items)));
            for (Index b = 0; b < num_blocks; ++b) {
                tz.expect_index("entity dimension");
                tz.expect_index("entity tag");
                const Index parametric = tz.expect_index("parametric flag");
                if (parametric != 0)
                    tz.error("parametric node blocks are not supported");
                const Index n = tz.expect_index("number of nodes in block");
                if (n < 0)
                    tz.error("negative node count in block");
                for (Index i = 0; i < n; ++i) {
                    const Index tag = tz.expect_index("node tag");
                    if (tag <= 0)
                        tz.error("node tag " + std::to_string(tag) + " is not positive");
                    const auto node = static_cast<Index>(node_coords.size() / 3) + i;
                    if (!node_of_tag.try_emplace(tag, node).second)
                        tz.error("duplicate node tag " + std::to_string(tag));
                }
                for (Index i = 0; i < n; ++i) {
                    node_coords.push_back(tz.expect_real("node x"));
                    node_coords.push_back(tz.expect_real("node y"));
                    node_coords.push_back(tz.expect_real("node z"));
                }
            }
            if (static_cast<Index>(node_coords.size() / 3) != num_nodes) {
                tz.error("$Nodes header announces " + std::to_string(num_nodes) +
                         " nodes, blocks contain " + std::to_string(node_coords.size() / 3));
            }
            tz.expect_keyword("$EndNodes");
            have_nodes = true;
        } else if (tok == "$Elements") {
            if (!have_nodes)
                tz.error("$Elements before $Nodes");
            if (have_elements)
                tz.error("more than one $Elements section");
            const Index num_blocks = tz.expect_index("number of element entity blocks");
            const Index num_elements = tz.expect_index("number of elements");
            tz.expect_index("minimum element tag");
            tz.expect_index("maximum element tag");
            if (num_blocks < 0 || num_elements < 0)
                tz.error("negative element count");
            corners.reserve(3 * static_cast<std::size_t>(std::min(num_elements, max_items)));
            Index total = 0;
            for (Index b = 0; b < num_blocks; ++b) {
                tz.expect_index("entity dimension");
                tz.expect_index("entity tag");
                const Index type = tz.expect_index("element type");
                const Index n = tz.expect_index("number of elements in block");
                if (n < 0)
                    tz.error("negative element count in block");
                Index nodes_per_element = 0;
                if (type == 2) {
                    nodes_per_element = 3;
                } else if (type == 15) {
                    nodes_per_element = 1;
                } else if (type == 1) {
                    nodes_per_element = 2;
                } else {
                    tz.error("element type " + std::to_string(type) + " (" +
                             gmsh_element_name(type) +
                             ") is not supported; only 3-node triangles (type 2) are imported, "
                             "points (15) and lines (1) are skipped");
                }
                for (Index e = 0; e < n; ++e) {
                    const Index elem_tag = tz.expect_index("element tag");
                    for (Index k = 0; k < nodes_per_element; ++k) {
                        const Index tag = tz.expect_index("element node tag");
                        if (type != 2)
                            continue;
                        const auto it = node_of_tag.find(tag);
                        if (it == node_of_tag.end()) {
                            tz.error("triangle " + std::to_string(elem_tag) +
                                     " references unknown node tag " + std::to_string(tag));
                        }
                        corners.push_back(it->second);
                    }
                }
                total += n;
            }
            if (total != num_elements) {
                tz.error("$Elements header announces " + std::to_string(num_elements) +
                         " elements, blocks contain " + std::to_string(total));
            }
            tz.expect_keyword("$EndElements");
            have_elements = true;
        } else if (!tok.empty() && tok.front() == '$') {
            // Unknown / unused section ($Entities, $PhysicalNames, $NodeData, ...): skip.
            const std::string end_tag = "$End" + std::string(tok.substr(1));
            const Index start_line = tz.line();
            std::string_view t;
            for (;;) {
                if (!tz.next(t)) {
                    fail(fn, path, start_line,
                         "section " + quote(tok) + " is not closed by " + quote(end_tag));
                }
                if (t == end_tag)
                    break;
            }
        } else {
            tz.error("unexpected token " + quote(tok) + " outside a section");
        }
    }
    if (!have_format)
        fail(fn, path, 0, "missing $MeshFormat section (not a Gmsh MSH file?)");
    if (!have_elements)
        fail(fn, path, 0, "missing $Elements section");

    // Keep only nodes referenced by triangles, in $Nodes order.
    const auto num_nodes = static_cast<Index>(node_coords.size() / 3);
    std::vector<Index> new_index(static_cast<std::size_t>(num_nodes), -1);
    for (const Index node : corners) new_index[static_cast<std::size_t>(node)] = 0;
    Index num_used = 0;
    for (Index& id : new_index) {
        if (id == 0)
            id = num_used++;
    }
    Vertices vertices(num_used, 3);
    for (Index node = 0; node < num_nodes; ++node) {
        const Index id = new_index[static_cast<std::size_t>(node)];
        if (id < 0)
            continue;
        for (Index c = 0; c < 3; ++c)
            vertices(id, c) = node_coords[static_cast<std::size_t>(3 * node + c)];
    }
    for (Index& node : corners) node = new_index[static_cast<std::size_t>(node)];
    if (num_used < num_nodes) {
        SBEM_DEBUG("read_gmsh: '{}': dropped {} node(s) not referenced by a triangle", path,
                   num_nodes - num_used);
    }
    return make_mesh(fn, path, std::move(vertices), to_triangles(corners));
}

void write_stl(const TriangleMesh& m, const std::string& path, StlFormat format) {
    require_triangles(m, "write_stl", path);
    if (format == StlFormat::binary) {
        write_stl_binary(m, path);
    } else {
        write_stl_ascii(m, path);
    }
}

void write_obj(const TriangleMesh& m, const std::string& path) {
    require_triangles(m, "write_obj", path);
    OutputFile out("write_obj", path);
    out.put("# SpeckleBem OBJ export, units: metres, ");
    out.put_index(m.num_vertices());
    out.put(" vertices, ");
    out.put_index(m.num_triangles());
    out.put(" triangles\n");
    const Vertices& v = m.vertices();
    for (Index i = 0; i < m.num_vertices(); ++i) {
        out.put("v ");
        out.put_real(v(i, 0));
        out.put(' ');
        out.put_real(v(i, 1));
        out.put(' ');
        out.put_real(v(i, 2));
        out.put('\n');
    }
    const Triangles& tri = m.triangles();
    for (Index t = 0; t < m.num_triangles(); ++t) {
        out.put('f');
        for (Index k = 0; k < 3; ++k) {
            out.put(' ');
            out.put_index(tri(t, k) + 1);
        }
        out.put('\n');
    }
    out.close();
}

void write_gmsh(const TriangleMesh& m, const std::string& path) {
    require_triangles(m, "write_gmsh", path);
    OutputFile out("write_gmsh", path);
    const Index nv = m.num_vertices();
    const Index nf = m.num_triangles();
    out.put("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n");

    // One surface entity (tag 1) with its bounding box, no physical tags, no curves.
    const auto [lo, hi] = m.bounding_box();
    out.put("$Entities\n0 0 1 0\n1");
    for (const Real x : {lo.x(), lo.y(), lo.z(), hi.x(), hi.y(), hi.z()}) {
        out.put(' ');
        out.put_real(x);
    }
    out.put(" 0 0\n$EndEntities\n");

    out.put("$Nodes\n1 ");
    out.put_index(nv);
    out.put(" 1 ");
    out.put_index(nv);
    out.put("\n2 1 0 ");
    out.put_index(nv);
    out.put('\n');
    for (Index i = 1; i <= nv; ++i) {
        out.put_index(i);
        out.put('\n');
    }
    const Vertices& v = m.vertices();
    for (Index i = 0; i < nv; ++i) {
        out.put_real(v(i, 0));
        out.put(' ');
        out.put_real(v(i, 1));
        out.put(' ');
        out.put_real(v(i, 2));
        out.put('\n');
    }
    out.put("$EndNodes\n");

    out.put("$Elements\n1 ");
    out.put_index(nf);
    out.put(" 1 ");
    out.put_index(nf);
    out.put("\n2 1 2 ");
    out.put_index(nf);
    out.put('\n');
    const Triangles& tri = m.triangles();
    for (Index t = 0; t < nf; ++t) {
        out.put_index(t + 1);
        for (Index k = 0; k < 3; ++k) {
            out.put(' ');
            out.put_index(tri(t, k) + 1);
        }
        out.put('\n');
    }
    out.put("$EndElements\n");
    out.close();
}

}  // namespace specklebem::geometry::io
