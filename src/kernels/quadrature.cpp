/// @file quadrature.cpp
/// Symmetric Dunavant rules on the triangle and Gauss-Legendre rules on [-1, 1].
///
/// Triangle rules: D.A. Dunavant, "High degree efficient symmetrical Gaussian quadrature
/// rules for the triangle", IJNME 21 (1985) 1129-1148, degrees 1..20. Each rule is stored
/// as a list of symmetry orbits of barycentric points:
///   * multiplicity 1: the centroid (1/3, 1/3, 1/3);
///   * multiplicity 3: the 3 distinct permutations of (1 - 2s, s, s), stored as (s, s);
///   * multiplicity 6: the 6 permutations of (1 - b - c, b, c), stored as (b, c);
/// with one weight per orbit (every point of an orbit carries that weight). Weights are
/// normalised so that they sum to 1 over the whole rule.
///
/// Provenance of the numbers: the generators and weights are Dunavant's published tables
/// (Dunavant 1985; 15 significant digits). Only numeric data, no code, was taken from
/// J. Burkardt's dunavant.f90, which reproduces those tables. Starting from the published
/// values, the moment equations sum_i w_i x_i^a y_i^b = 2 a! b! / (a + b + 2)!,
/// a + b <= degree, were re-solved by Newton iteration in 80-digit arithmetic with the orbit
/// structure (multiplicities and symmetry) held fixed; the results are written below with 21
/// significant digits. They differ from the published values by at most 6.4e-16 for degrees
/// 1-19 and by 3.6e-15 for degree 20 (the 3-orbit s = 0.50095..., whose published value
/// 0.5009504643522 is given with only 13 decimals), so they are the same rules, now exact to
/// double precision instead of to ~1e-15. All 20 degrees use the Dunavant tables; no fallback
/// construction is needed.
///
/// Point counts: 1, 3, 4, 6, 7, 12, 13, 16, 19, 25, 27, 33, 37, 42, 48, 52, 61, 70, 73, 79.
///
/// Rules that are not "PI" (positive weights, all points strictly inside the triangle;
/// see triangle_rule_is_positive_interior()):
///   * degree  3: negative centroid weight (-0.5625).
///   * degree  7: negative centroid weight (-0.1496).
///   * degree 11: one 3-orbit lies outside the triangle (barycentric coordinate -0.0692).
///   * degree 15: one 3-orbit lies outside the triangle (barycentric coordinate -0.0139).
///   * degree 16: one 6-orbit lies outside the triangle (barycentric coordinate -0.0043).
///   * degree 18: one 6-orbit lies outside (barycentric coordinates -0.0352 and 1.0143) and
///                one 3-orbit has a negative weight (-0.0022).
///   * degree 20: one 3-orbit (barycentric coordinate -0.0019) and one 6-orbit (barycentric
///                coordinate -0.0084) lie outside, and one 3-orbit has a negative weight
///                (-0.0006).
/// Points outside the triangle are harmless for the smooth integrands used in the operator
/// assembly (the integrand is a polynomial or an analytic function of the position), but
/// callers that evaluate data defined only on the triangle should prefer the other degrees.
///
/// Gauss-Legendre: Newton iteration on the three-term Legendre recurrence, started from
/// Tricomi's asymptotic approximation of the roots; O(n^2) work. Checked against a 40-digit
/// reference for n = 100, 300, 600: nodes within 6e-17 and weights within 1.2e-16 absolute
/// (the tiny end-point weights are relatively less accurate, ~1e-12 for n = 300, because
/// 1 - x^2 is formed from the rounded node).
#include "specklebem/kernels/quadrature.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace specklebem::kernels {

namespace {

/// One symmetry orbit of a Dunavant rule (see the file comment).
struct Orbit {
    int multiplicity;  ///< 1, 3 or 6
    Real b;            ///< second barycentric coordinate of the generator
    Real c;            ///< third barycentric coordinate of the generator
    Real w;            ///< weight of every point of the orbit
};

constexpr Real kThird = 1.0 / 3.0;

// Degree 1: 1 point.
constexpr std::array<Orbit, 1> kDegree1{{
    {1, kThird, kThird, 1.00000000000000000000e+0},
}};

// Degree 2: 3 points.
constexpr std::array<Orbit, 1> kDegree2{{
    {3, 1.66666666666666666667e-1, 1.66666666666666666667e-1, 3.33333333333333333333e-1},
}};

// Degree 3: 4 points.
constexpr std::array<Orbit, 2> kDegree3{{
    {1, kThird, kThird, -5.62500000000000000000e-1},
    {3, 2.00000000000000000000e-1, 2.00000000000000000000e-1, 5.20833333333333333333e-1},
}};

// Degree 4: 6 points.
constexpr std::array<Orbit, 2> kDegree4{{
    {3, 4.45948490915964886318e-1, 4.45948490915964886318e-1, 2.23381589678011465695e-1},
    {3, 9.15762135097707434596e-2, 9.15762135097707434596e-2, 1.09951743655321867638e-1},
}};

// Degree 5: 7 points.
constexpr std::array<Orbit, 3> kDegree5{{
    {1, kThird, kThird, 2.25000000000000000000e-1},
    {3, 4.70142064105115089770e-1, 4.70142064105115089770e-1, 1.32394152788506180738e-1},
    {3, 1.01286507323456338801e-1, 1.01286507323456338801e-1, 1.25939180544827152596e-1},
}};

// Degree 6: 12 points.
constexpr std::array<Orbit, 3> kDegree6{{
    {3, 2.49286745170910421292e-1, 2.49286745170910421292e-1, 1.16786275726379366025e-1},
    {3, 6.30890144915022283403e-2, 6.30890144915022283403e-2, 5.08449063702068169209e-2},
    {6, 3.10352451033784405417e-1, 6.36502499121398647230e-1, 8.28510756183735751936e-2},
}};

// Degree 7: 13 points.
constexpr std::array<Orbit, 4> kDegree7{{
    {1, kThird, kThird, -1.49570044467681750630e-1},
    {3, 2.60345966079039826926e-1, 2.60345966079039826926e-1, 1.75615257433207811754e-1},
    {3, 6.51301029022158115380e-2, 6.51301029022158115380e-2, 5.33472356088384912700e-2},
    {6, 3.12865496004873861407e-1, 6.38444188569809726800e-1, 7.71137608902571402599e-2},
}};

// Degree 8: 16 points.
constexpr std::array<Orbit, 5> kDegree8{{
    {1, kThird, kThird, 1.44315607677787168251e-1},
    {3, 4.59292588292723156029e-1, 4.59292588292723156029e-1, 9.50916342672846247939e-2},
    {3, 1.70569307751760206622e-1, 1.70569307751760206622e-1, 1.03217370534718250282e-1},
    {3, 5.05472283170309754584e-2, 5.05472283170309754584e-2, 3.24584976231980803109e-2},
    {6, 2.63112829634638113422e-1, 7.28492392955404281241e-1, 2.72303141744349942648e-2},
}};

// Degree 9: 19 points.
constexpr std::array<Orbit, 6> kDegree9{{
    {1, kThird, kThird, 9.71357962827988338192e-2},
    {3, 4.89682519198737627784e-1, 4.89682519198737627784e-1, 3.13347002271390705369e-2},
    {3, 4.37089591492936637270e-1, 4.37089591492936637270e-1, 7.78275410047742793167e-2},
    {3, 1.88203535619032730241e-1, 1.88203535619032730241e-1, 7.96477389272102530329e-2},
    {3, 4.47295133944527098651e-2, 4.47295133944527098651e-2, 2.55776756586980312617e-2},
    {6, 2.21962989160765695675e-1, 7.41198598784498020690e-1, 4.32835393772893772894e-2},
}};

// Degree 10: 25 points.
constexpr std::array<Orbit, 6> kDegree10{{
    {1, kThird, kThird, 9.08179903827535800953e-2},
    {3, 4.85577633383657377368e-1, 4.85577633383657377368e-1, 3.67259577564667047170e-2},
    {3, 1.09481575485037054795e-1, 1.09481575485037054795e-1, 4.53210594355279347826e-2},
    {6, 3.07939838764120950165e-1, 5.50352941820999095078e-1, 7.27579168454201086043e-2},
    {6, 2.46672560639902693917e-1, 7.28323904597410920009e-1, 2.83272425310574848367e-2},
    {6, 6.68032510122002657735e-2, 9.23655933587500276646e-1, 9.42166696373282345993e-3},
}};

// Degree 11: 27 points.
constexpr std::array<Orbit, 7> kDegree11{{
    {3, 5.34611048270758309359e-1, 5.34611048270758309359e-1, 9.27006328960676050660e-4},
    {3, 3.98969302965855222611e-1, 3.98969302965855222611e-1, 7.71495349148131228615e-2},
    {3, 2.03309900431282473351e-1, 2.03309900431282473351e-1, 5.93229773807740731158e-2},
    {3, 1.19350912282581309581e-1, 1.19350912282581309581e-1, 3.61845405034180793383e-2},
    {3, 3.23649481112758931588e-2, 3.23649481112758931588e-2, 1.36597310026778619587e-2},
    {6, 3.56620648261292582437e-1, 5.93201213428212752489e-1, 5.23371119622040711171e-2},
    {6, 1.71488980304041549710e-1, 8.07489003159792153167e-1, 2.07076596391406888870e-2},
}};

// Degree 12: 33 points.
constexpr std::array<Orbit, 8> kDegree12{{
    {3, 4.88217389773804882565e-1, 4.88217389773804882565e-1, 2.57310664404553354178e-2},
    {3, 4.39724392294460272980e-1, 4.39724392294460272980e-1, 4.36925445380384021355e-2},
    {3, 2.71210385012115922346e-1, 2.71210385012115922346e-1, 6.28582242178851003543e-2},
    {3, 1.27576145541585924674e-1, 1.27576145541585924674e-1, 3.47961129307089429893e-2},
    {3, 2.13173504532103702469e-2, 2.13173504532103702469e-2, 6.16626105155901723387e-3},
    {6, 2.75713269685514193975e-1, 6.08943235779787806856e-1, 4.03715577663809295178e-2},
    {6, 2.81325580989939548248e-1, 6.95836086787803422142e-1, 2.23567732023034457118e-2},
    {6, 1.16251915907597141241e-1, 8.58014033544072630591e-1, 1.73162311086588923716e-2},
}};

// Degree 13: 37 points.
constexpr std::array<Orbit, 10> kDegree13{{
    {1, kThird, kThird, 5.25209234008019981555e-2},
    {3, 4.95048184939704628012e-1, 4.95048184939704628012e-1, 1.12801452093295573309e-2},
    {3, 4.68716635109573949985e-1, 4.68716635109573949985e-1, 3.14235183624542375838e-2},
    {3, 4.14521336801276711510e-1, 4.14521336801276711510e-1, 4.70725025041943416627e-2},
    {3, 2.29399572042831220054e-1, 2.29399572042831220054e-1, 4.73635865363546011088e-2},
    {3, 1.14424495196330145703e-1, 1.14424495196330145703e-1, 3.11675290457938407983e-2},
    {3, 2.48113913634589652368e-2, 2.48113913634589652368e-2, 7.97577146507436291359e-3},
    {6, 2.68794997058761106005e-1, 6.36351174561660365304e-1, 3.68484027287322183385e-2},
    {6, 2.91730066734287723986e-1, 6.90169159986905372579e-1, 1.74014633038220452721e-2},
    {6, 1.26357385491668692162e-1, 8.51409537834241191542e-1, 1.55217868390449326645e-2},
}};

// Degree 14: 42 points.
constexpr std::array<Orbit, 10> kDegree14{{
    {3, 4.88963910362178638677e-1, 4.88963910362178638677e-1, 2.18835813694288906408e-2},
    {3, 4.17644719340453922509e-1, 4.17644719340453922509e-1, 3.27883535441253506413e-2},
    {3, 2.73477528308838659755e-1, 2.73477528308838659755e-1, 5.17741045072915863148e-2},
    {3, 1.77205532412543436957e-1, 1.77205532412543436957e-1, 4.21625887369930175382e-2},
    {3, 6.17998830908726012675e-2, 6.17998830908726012675e-2, 1.44336996697766676017e-2},
    {3, 1.93909612487010481783e-2, 1.93909612487010481783e-2, 4.92340360240008168183e-3},
    {6, 1.72266687821355578375e-1, 7.70608554774996482589e-1, 2.46657532125636739629e-2},
    {6, 3.36861459796345001744e-1, 5.70222290846683173498e-1, 3.85715107870606832285e-2},
    {6, 2.98372882136257752971e-1, 6.86980167808087837359e-1, 1.44363081135338404961e-2},
    {6, 1.18974497696956845398e-1, 8.79757171370171129515e-1, 5.01022883850067176986e-3},
}};

// Degree 15: 48 points.
constexpr std::array<Orbit, 11> kDegree15{{
    {3, 5.06972916858242961717e-1, 5.06972916858242961717e-1, 1.91687564284861784865e-3},
    {3, 4.31406354283022630471e-1, 4.31406354283022630471e-1, 4.42490272711447276221e-2},
    {3, 2.77693644847144404367e-1, 2.77693644847144404367e-1, 5.11865487188521183149e-2},
    {3, 1.26464891041253862399e-1, 1.26464891041253862399e-1, 2.36877358706878071654e-2},
    {3, 7.08083859746859007602e-2, 7.08083859746859007602e-2, 1.32897756900205202205e-2},
    {3, 1.89651702410733385725e-2, 1.89651702410733385725e-2, 4.74891660819184758912e-3},
    {6, 2.61311371140087457523e-1, 6.04954466893291458068e-1, 3.85500725995925154006e-2},
    {6, 3.88046767090268815679e-1, 5.75586555512814358874e-1, 2.72158143206242685931e-2},
    {6, 2.85712220049916009542e-1, 7.24462663076654659385e-1, 2.18207736679702888298e-3},
    {6, 2.15599664072284068521e-1, 7.47556466051837739896e-1, 2.15053198477313631070e-2},
    {6, 1.03575616576385751527e-1, 8.83964574092415537857e-1, 7.67394263104867130274e-3},
}};

// Degree 16: 52 points.
constexpr std::array<Orbit, 13> kDegree16{{
    {1, kThird, kThird, 4.68756974276416469759e-2},
    {3, 4.97380541948438400922e-1, 4.97380541948438400922e-1, 6.40587857858496736404e-3},
    {3, 4.13469438549352377309e-1, 4.13469438549352377309e-1, 4.17102967393868588108e-2},
    {3, 4.70458599066991334193e-1, 4.70458599066991334193e-1, 2.68914842500644245929e-2},
    {3, 2.40553749969520885187e-1, 2.40553749969520885187e-1, 4.21325227616496511518e-2},
    {3, 1.47965794222572804586e-1, 1.47965794222572804586e-1, 3.00002668427729839022e-2},
    {3, 7.54651876574741750153e-2, 7.54651876574741750153e-2, 1.42000989250241858596e-2},
    {3, 1.65964026230249240480e-2, 1.65964026230249240480e-2, 3.58246235127336770590e-3},
    {6, 2.96555596579887400122e-1, 5.99868711174860569655e-1, 3.27731474606274158382e-2},
    {6, 3.37723063403079120895e-1, 6.42193524941504994099e-1, 1.52983062484411806151e-2},
    {6, 2.04748281642812075713e-1, 7.99592720971326707395e-1, 2.38624419283868600949e-3},
    {6, 1.89358492130622511754e-1, 7.68699721401367647965e-1, 1.90847927558988993897e-2},
    {6, 8.52836156826572236654e-2, 9.00399064086661410553e-1, 6.85005454654199062457e-3},
}};

// Degree 17: 61 points.
constexpr std::array<Orbit, 15> kDegree17{{
    {1, kThird, kThird, 3.34371992908029305711e-2},
    {3, 4.97170540556773964215e-1, 4.97170540556773964215e-1, 5.09341544050679988681e-3},
    {3, 4.82176322624624698167e-1, 4.82176322624624698167e-1, 1.46708645276379984138e-2},
    {3, 4.50239969020781802570e-1, 4.50239969020781802570e-1, 2.43508783536722789761e-2},
    {3, 4.00266239377396971039e-1, 4.00266239377396971039e-1, 3.11075508689694771529e-2},
    {3, 2.52141267970952566282e-1, 2.52141267970952566282e-1, 3.12571112186204813326e-2},
    {3, 1.62047004658461592158e-1, 1.62047004658461592158e-1, 2.48156543396648410965e-2},
    {3, 7.58758822607457065099e-2, 7.58758822607457065099e-2, 1.40560730705570384283e-2},
    {3, 1.56547269678218232141e-2, 1.56547269678218232141e-2, 3.19467617377882721947e-3},
    {6, 3.34319867363657914537e-1, 6.55493203809423124391e-1, 8.11965531899247538870e-3},
    {6, 2.92221537796943827283e-1, 5.72337590532020083519e-1, 2.68057422831625105569e-2},
    {6, 3.19574885423189621300e-1, 6.26001190286227972641e-1, 1.84599932108220817228e-2},
    {6, 1.90704224192291820727e-1, 7.96427214974071324823e-1, 8.47686853432842749244e-3},
    {6, 1.80483211648746181130e-1, 7.52351005937729239198e-1, 1.82927967700248495862e-2},
    {6, 8.07113136795638452298e-2, 9.04625504095607917610e-1, 6.66563200416529557128e-3},
}};

// Degree 18: 70 points.
constexpr std::array<Orbit, 17> kDegree18{{
    {1, kThird, kThird, 3.08099399376476285107e-2},
    {3, 4.93344808630921311149e-1, 4.93344808630921311149e-1, 9.07243667940440080038e-3},
    {3, 4.69210594241956654806e-1, 4.69210594241956654806e-1, 1.87613169395937514969e-2},
    {3, 4.36281395887005580101e-1, 4.36281395887005580101e-1, 1.94410979854770960962e-2},
    {3, 3.94846170673416629531e-1, 3.94846170673416629531e-1, 2.77539486108096408425e-2},
    {3, 2.49794568803156885831e-1, 2.49794568803156885831e-1, 3.22562253514573114192e-2},
    {3, 1.61432193743842599795e-1, 1.61432193743842599795e-1, 2.50740326169219906142e-2},
    {3, 7.65982274853713836960e-2, 7.65982274853713836960e-2, 1.52719279718316920167e-2},
    {3, 2.42524393534500026068e-2, 2.42524393534500026068e-2, 6.79392202296295883847e-3},
    {3, 4.31463672169650277595e-2, 4.31463672169650277595e-2, -2.22309872992044468667e-3},
    {6, 3.58911494940944058779e-1, 6.32657968856635782825e-1, 6.33191407640588602279e-3},
    {6, 2.94402476751956562272e-1, 5.74410971510855265553e-1, 2.72575380491384614359e-2},
    {6, 3.25017801641813683117e-1, 6.24779046792511821564e-1, 1.76767856494646227214e-2},
    {6, 1.84737559666046382805e-1, 7.48933176523037346346e-1, 1.83794846380700464704e-2},
    {6, 2.18796800013320796872e-1, 7.69207005420443416823e-1, 8.10473280819189409580e-3},
    {6, 1.01179597136407545148e-1, 8.83962302273467016263e-1, 7.63412907072451636977e-3},
    {6, 2.08747552825861923171e-2, 1.01434726000536279787e+0, 4.61876607941027464884e-5},
}};

// Degree 19: 73 points.
constexpr std::array<Orbit, 17> kDegree19{{
    {1, kThird, kThird, 3.29063313889186506551e-2},
    {3, 4.89609987073006328637e-1, 4.89609987073006328637e-1, 1.03307318912720539310e-2},
    {3, 4.54536892697892659863e-1, 4.54536892697892659863e-1, 2.23872472630163924977e-2},
    {3, 4.01416680649431185943e-1, 4.01416680649431185943e-1, 3.02661258694680698404e-2},
    {3, 2.55551654403097612686e-1, 2.55551654403097612686e-1, 3.04909678021977799234e-2},
    {3, 1.77077942152129552935e-1, 1.77077942152129552935e-1, 2.41592127416409045926e-2},
    {3, 1.10061053227951859424e-1, 1.10061053227951859424e-1, 1.60508035868008760334e-2},
    {3, 5.55286242518396735611e-2, 5.55286242518396735611e-2, 8.08458026178406068560e-3},
    {3, 1.26218637772286682857e-2, 1.26218637772286682857e-2, 2.07936202748478068412e-3},
    {6, 3.95754787356942878571e-1, 6.00633794794644981621e-1, 3.88487690498139090143e-3},
    {6, 3.07929983880436245053e-1, 5.57603261588783965888e-1, 2.55741606120219034494e-2},
    {6, 2.64566948406520211700e-1, 7.20987025817365052084e-1, 8.88090357333805807699e-3},
    {6, 3.58539352205950580841e-1, 5.94527068955870927813e-1, 1.61245467617313914229e-2},
    {6, 1.57807405968594744756e-1, 8.39331473680838584938e-1, 2.49194181749067378203e-3},
    {6, 7.50505969759109437200e-2, 7.01087978926173369098e-1, 1.82428401189505785312e-2},
    {6, 1.42421601113383443602e-1, 8.22931324069856632483e-1, 1.02585637361985220773e-2},
    {6, 6.54946280829377020913e-2, 9.24344252620784029780e-1, 3.79992885530191422208e-3},
}};

// Degree 20: 79 points.
constexpr std::array<Orbit, 19> kDegree20{{
    {1, kThird, kThird, 3.30570555416239332652e-2},
    {3, 5.00950464352196437331e-1, 5.00950464352196437331e-1, 8.67019185663450772327e-4},
    {3, 4.88212957934727422798e-1, 4.88212957934727422798e-1, 1.16600527164471314372e-2},
    {3, 4.55136681950281953095e-1, 4.55136681950281953095e-1, 2.28769363564207456155e-2},
    {3, 4.01996259318289133750e-1, 4.01996259318289133750e-1, 3.04489826739380009382e-2},
    {3, 2.55892909759421188062e-1, 2.55892909759421188062e-1, 3.06248917253546795810e-2},
    {3, 1.76488255995105999229e-1, 1.76488255995105999229e-1, 2.43680576768004247093e-2},
    {3, 1.04170855336758434604e-1, 1.04170855336758434604e-1, 1.59974320320239237948e-2},
    {3, 5.30689638409299281261e-2, 5.30689638409299281261e-2, 7.69830181560230745003e-3},
    {3, 4.16187151960289032075e-2, 4.16187151960289032075e-2, -6.32060497487575047564e-4},
    {3, 1.15819214068221660748e-2, 1.15819214068221660748e-2, 1.75113430119276650897e-3},
    {6, 3.44855770229001099705e-1, 6.06402646106159554437e-1, 1.64658391895757595889e-2},
    {6, 3.77843269594854033384e-1, 6.15842614456540763207e-1, 4.83903354048480452612e-3},
    {6, 3.06635479062356759882e-1, 5.59048000390295468301e-1, 2.58049065346500153397e-2},
    {6, 2.49419362774742182432e-1, 7.36606743262865665272e-1, 8.47109105444064728032e-3},
    {6, 2.12775724802801620448e-1, 7.11675142287434223175e-1, 1.83549141062797474919e-2},
    {6, 1.46965436053239127908e-1, 8.61402717154987493104e-1, 7.04404677908216451446e-4},
    {6, 1.37726978828923138653e-1, 8.35586957912362784266e-1, 1.01126849274619006083e-2},
    {6, 5.96961091490065434994e-2, 9.29756171556852617627e-1, 3.57390938595032528917e-3},
}};

constexpr int kMaxTriangleDegree = 20;

/// Documented point counts of the Dunavant rules, degree 1..20.
constexpr std::array<std::size_t, kMaxTriangleDegree> kPointCounts = {
    1, 3, 4, 6, 7, 12, 13, 16, 19, 25, 27, 33, 37, 42, 48, 52, 61, 70, 73, 79};

/// Expands the orbits of the rule of degree `degree` into explicit barycentric points.
/// Throws std::logic_error if the table is inconsistent (invalid multiplicity or a point
/// count that differs from kPointCounts).
TriangleRule expand_orbits(int degree, std::span<const Orbit> orbits) {
    const std::size_t expected = kPointCounts[static_cast<std::size_t>(degree - 1)];
    std::size_t count = 0;
    for (const Orbit& o : orbits) {
        if (o.multiplicity != 1 && o.multiplicity != 3 && o.multiplicity != 6) {
            throw std::logic_error("triangle_rule: invalid orbit multiplicity " +
                                   std::to_string(o.multiplicity) + " in the degree-" +
                                   std::to_string(degree) + " table");
        }
        count += static_cast<std::size_t>(o.multiplicity);
    }
    if (count != expected) {
        throw std::logic_error("triangle_rule: degree-" + std::to_string(degree) + " table has " +
                               std::to_string(count) + " points, expected " +
                               std::to_string(expected));
    }

    TriangleRule rule;
    rule.barycentric.reserve(count);
    rule.weights.reserve(count);
    for (const Orbit& o : orbits) {
        const Real a = 1.0 - o.b - o.c;
        switch (o.multiplicity) {
            case 1:
                rule.barycentric.emplace_back(kThird, kThird, kThird);
                break;
            case 3:
                // Generator (a, s, s) with s = o.b = o.c: the three distinct permutations.
                rule.barycentric.emplace_back(a, o.b, o.c);
                rule.barycentric.emplace_back(o.b, a, o.c);
                rule.barycentric.emplace_back(o.b, o.c, a);
                break;
            default:  // 6, validated above
                // Generator (a, b, c) with distinct entries: all six permutations.
                rule.barycentric.emplace_back(a, o.b, o.c);
                rule.barycentric.emplace_back(a, o.c, o.b);
                rule.barycentric.emplace_back(o.b, a, o.c);
                rule.barycentric.emplace_back(o.b, o.c, a);
                rule.barycentric.emplace_back(o.c, a, o.b);
                rule.barycentric.emplace_back(o.c, o.b, a);
                break;
        }
        rule.weights.insert(rule.weights.end(), static_cast<std::size_t>(o.multiplicity), o.w);
    }
    return rule;
}

std::array<TriangleRule, kMaxTriangleDegree> build_triangle_rules() {
    return {
        expand_orbits(1, kDegree1),   expand_orbits(2, kDegree2),   expand_orbits(3, kDegree3),
        expand_orbits(4, kDegree4),   expand_orbits(5, kDegree5),   expand_orbits(6, kDegree6),
        expand_orbits(7, kDegree7),   expand_orbits(8, kDegree8),   expand_orbits(9, kDegree9),
        expand_orbits(10, kDegree10), expand_orbits(11, kDegree11), expand_orbits(12, kDegree12),
        expand_orbits(13, kDegree13), expand_orbits(14, kDegree14), expand_orbits(15, kDegree15),
        expand_orbits(16, kDegree16), expand_orbits(17, kDegree17), expand_orbits(18, kDegree18),
        expand_orbits(19, kDegree19), expand_orbits(20, kDegree20)};
}

/// Legendre polynomial P_n(x) and its derivative by the three-term recurrence.
std::pair<Real, Real> legendre_and_derivative(int n, Real x) {
    Real p0 = 1.0;  // P_{k-1}
    Real p1 = x;    // P_k
    for (int k = 1; k < n; ++k) {
        const Real kr = static_cast<Real>(k);
        const Real p2 = ((2.0 * kr + 1.0) * x * p1 - kr * p0) / (kr + 1.0);
        p0 = p1;
        p1 = p2;
    }
    // P_n'(x) = n (x P_n - P_{n-1}) / (x^2 - 1); |x| < 1 strictly for interior roots.
    const Real dp = static_cast<Real>(n) * (x * p1 - p0) / (x * x - 1.0);
    return {p1, dp};
}

}  // namespace

const TriangleRule& triangle_rule(int degree) {
    if (degree < 1 || degree > kMaxTriangleDegree) {
        throw std::invalid_argument("triangle_rule: degree must be in 1..20, got " +
                                    std::to_string(degree));
    }
    // Function-local static: built once, initialisation is thread-safe (C++11 and later).
    static const std::array<TriangleRule, kMaxTriangleDegree> rules = build_triangle_rules();
    return rules[static_cast<std::size_t>(degree - 1)];
}

bool triangle_rule_is_positive_interior(int degree) {
    const TriangleRule& rule = triangle_rule(degree);  // validates the degree
    for (std::size_t i = 0; i < rule.weights.size(); ++i) {
        if (!(rule.weights[i] > 0.0) || !(rule.barycentric[i].minCoeff() > 0.0)) {
            return false;
        }
    }
    return true;
}

LineRule gauss_legendre(int n) {
    if (n < 1) {
        throw std::invalid_argument("gauss_legendre: number of points must be >= 1, got " +
                                    std::to_string(n));
    }
    const auto un = static_cast<std::size_t>(n);
    LineRule rule;
    rule.nodes.assign(un, 0.0);
    rule.weights.assign(un, 0.0);

    const Real nr = static_cast<Real>(n);
    const Real eps = std::numeric_limits<Real>::epsilon();
    // Roots are symmetric: compute the positive ones (i = 1..ceil(n/2), descending in x),
    // mirror them, and set the middle root of an odd rule to exactly zero.
    const int half = n / 2 + n % 2;  // ceil(n / 2) without overflow at INT_MAX
    for (int i = 1; i <= half; ++i) {
        // Tricomi: x_i ~ (1 - 1/(8n^2) + 1/(8n^3)) cos(pi (4i - 1) / (4n + 2)).
        const Real theta = constants::pi * (4.0 * static_cast<Real>(i) - 1.0) / (4.0 * nr + 2.0);
        Real x = (1.0 - 1.0 / (8.0 * nr * nr) + 1.0 / (8.0 * nr * nr * nr)) * std::cos(theta);
        const bool middle = (n % 2 == 1) && (i == half);
        if (middle) {
            x = 0.0;
        } else {
            // Newton until the update is at the rounding level, then one polishing step.
            for (int it = 0; it < 100; ++it) {
                const auto [p, dp] = legendre_and_derivative(n, x);
                const Real dx = p / dp;
                x -= dx;
                if (std::abs(dx) <= 4.0 * eps * std::abs(x)) {
                    break;
                }
            }
            const auto [p, dp] = legendre_and_derivative(n, x);
            x -= p / dp;
        }
        const Real dp = legendre_and_derivative(n, x).second;
        const Real w = 2.0 / ((1.0 - x * x) * dp * dp);
        const auto lo = static_cast<std::size_t>(i - 1);  // index of -x (ascending order)
        const std::size_t hi = un - 1 - lo;               // index of +x
        rule.nodes[lo] = -x;
        rule.nodes[hi] = x;
        rule.weights[lo] = w;
        rule.weights[hi] = w;
    }
    return rule;
}

}  // namespace specklebem::kernels
