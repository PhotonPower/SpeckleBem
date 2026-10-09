# References

## Primary
- **Fu L., Daiber-Huppert M., Frenner K., Osten W.** *Simulation of realistic speckle fields by using surface integral equation and multi-level fast multipole method.* Opt. Lasers Eng. 162 (2023) 107438. https://doi.org/10.1016/j.optlaseng.2022.107438 — the design reference for this project (formulation choice, mesh sizes, validation setup, speckle statistics).
- Fu L., Frenner K., Osten W. *Rigorous speckle simulation using surface integral equations and higher order boundary element method.* Opt. Lett. 39 (2014) 4104.

## Surface integral equations and formulations
- Stratton J.A., Chu L.J. *Diffraction theory of electromagnetic waves.* Phys. Rev. 56 (1939) 99.
- Medgyesi-Mitschang L.N., Putnam J.M., Gedera M.B. *Generalized method of moments for three-dimensional penetrable scatterers.* JOSA A 11 (1994) 1383. (PMCHWT, tangential operators)
- Rao S.M., Wilton D.R., Glisson A.W. *Electromagnetic scattering by surfaces of arbitrary shape.* IEEE TAP 30 (1982) 409. (RWG)
- Solís D.M., Taboada J.M., Rubiños-López O., Obelleiro F. *Improved combined tangential formulation for electromagnetic analysis of penetrable bodies.* JOSA B 32 (2015) 1780. (ICTF)
- Karaosmanoğlu B., Yılmaz A., Ergül Ö. *A comparative study of surface integral equations for accurate and efficient analysis of plasmonic structures.* IEEE TAP 65 (2017) 3049. (MCTF)
- Karaosmanoğlu B., Ergül Ö. ACES Journal 34(5) (2019) 811. (MCTF matrix form, Eq. 1; confirms Table 1, issue #15)
- Ergül Ö., Gürel L. *Comparison of integral-equation formulations ... with the multilevel fast multipole algorithm.* IEEE TAP 57 (2009) 176. (JMCFIE and others)
- Gómez-Sousa H., Rubiños-López O., Martínez-Lorenzo J.Á. *Comparison of iterative solvers for electromagnetic analysis of plasmonic nanostructures using multiple surface integral equation formulations.* J. Electromagn. Waves Appl. 30 (2016) 456.

## Singular integrals
- Hänninen I., Taskinen M., Sarvas J. *Singularity subtraction integral formulae for surface integral equations with RWG, rooftop and hybrid basis functions.* PIER 63 (2006) 243.
- Wilton D.R. et al. *Potential integrals for uniform and linear source distributions on polygonal and polyhedral domains.* IEEE TAP 32 (1984) 276.
- Graglia R.D. *On the numerical integration of the linear shape functions times the 3-D Green's function or its gradient on a plane triangle.* IEEE TAP 41 (1993) 1448.
- Dunavant D.A. *High degree efficient symmetrical Gaussian quadrature rules for the triangle.* IJNME 21 (1985) 1129.
- Van Oosterom A., Strackee J. *The solid angle of a plane triangle.* IEEE Trans. Biomed. Eng. 30 (1983) 125. (solid-angle term of the static gradient integral)

## Fast multipole method
- Greengard L., Rokhlin V. *A fast algorithm for particle simulations.* J. Comput. Phys. 73 (1987) 325.
- Song J., Lu C.-C., Chew W.C. *Multilevel fast multipole algorithm for electromagnetic scattering by large complex objects.* IEEE TAP 45 (1997) 1488.
- Sheng X.-Q., Jin J.-M., Song J., Chew W.C., Lu C.-C. *Solution of combined-field integral equation using MLFMA for scattering by homogeneous bodies.* IEEE TAP 46 (1998) 1718.
- Chew W.C., Jin J.-M., Michielssen E., Song J. (eds.) *Fast and Efficient Algorithms in Computational Electromagnetics.* Artech House 2001.
- Gumerov N.A., Duraiswami R., Borovikova E.A. *Data structures, optimal choice of parameters, and complexity results for generalized multilevel fast multipole methods in d dimensions.* UMIACS TR 2003.
- Ergül Ö., Gürel L. *The Multilevel Fast Multipole Algorithm (MLFMA) for Solving Large-Scale Computational Electromagnetics Problems.* Wiley/IEEE 2014.
- Ergül Ö., Malas T., Gürel L. *Analysis of dielectric photonic-crystal problems with MLFMA and Schur-complement preconditioners.* J. Lightwave Technol. 29 (2011) 888.
- Daiber-Huppert M. *Open source implementation of the generalized Mie theory, accelerated by the multilevel FMM.* Master thesis, Univ. Stuttgart 2020.

## Other compression methods
- Bebendorf M. *Approximation of boundary element matrices.* Numer. Math. 86 (2000) 565. (ACA)
- Zhao K., Vouvakis M.N., Lee J.-F. *The adaptive cross approximation algorithm for accelerated method of moments computations of EMC problems.* IEEE TEMC 47 (2005) 763.
- Hackbusch W. *Hierarchical Matrices: Algorithms and Analysis.* Springer 2015.
- Börm S. *Efficient Numerical Methods for Non-local Operators: H²-Matrices.* EMS 2010.
- Michielssen E., Boag A. *A multilevel matrix decomposition algorithm for analyzing scattering from large structures.* IEEE TAP 44 (1996) 1086. (MLMDA / butterfly)
- Guo H., Liu Y., Hu J., Michielssen E. *A butterfly-based direct integral-equation solver using hierarchical LU factorization.* IEEE TAP 65 (2017) 4742.

## Solvers
- Saad Y., Schultz M.H. *GMRES.* SIAM J. Sci. Stat. Comput. 7 (1986) 856.
- Frayssé V., Giraud L., Gratton S., Langou J. *Algorithm 842: A set of GMRES routines for real and complex arithmetics.* ACM TOMS 31 (2005) 228.

## Speckle and rough-surface scattering
- Goodman J.W. *Speckle Phenomena in Optics: Theory and Applications.* 2nd ed., SPIE 2020.
- Goodman J.W. *Some properties of speckle from smooth objects.* Opt. Eng. 49 (2010) 068001.
- Ruffing B. *Application of speckle-correlation methods to surface-roughness measurement.* JOSA A 3 (1986) 1297.
- Persson U. *Surface roughness measurement on machined surfaces using angular speckle correlation.* J. Mater. Process. Technol. 180 (2006) 233.
- Simonsen I. *Optics of surface disordered systems.* Eur. Phys. J. Spec. Top. 181 (2010) 1.
- Warnick K.F., Chew W.C. *Numerical simulation methods for rough surface scattering.* Waves Random Media 11 (2001) R1.

## Mie theory
- Bohren C.F., Huffman D.R. *Absorption and Scattering of Light by Small Particles.* Wiley 1983.
- Wiscombe W.J. *Improved Mie scattering algorithms.* Appl. Opt. 19 (1980) 1505.

## Material data
- Johnson P.B., Christy R.W. *Optical constants of the noble metals.* Phys. Rev. B 6 (1972) 4370.
- Aspnes D.E., Studna A.A. *Dielectric functions and optical parameters of Si, Ge, GaP, GaAs, GaSb, InP, InAs, and InSb from 1.5 to 6.0 eV.* Phys. Rev. B 27 (1983) 985.
- https://refractiveindex.info
