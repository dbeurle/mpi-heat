/////////////////////////////////////////////////////////////////////////////
//
//
// Problem:		dT/dt = alpha_heat * Grad^2 T
//
// Method:		Finite Element Method with linear 3D tetrahedral
// elements
//				and Implicit Euler Method and Conjugate Gradient
// Method
//
// Allocate 1 Node on Avoca:    salloc -N 1 -t1:00:00 --account VR0084
//
// Compilation on Avoca:        mpixlcxx -qsmp=omp (-qMAXMEM=10000) *.cpp
//
// Execution on avoca:          srun -n(numberOfSubgrids) --nodes
// (numberOfNodes) --ntasks-per-node (2,4,8,16 etc.) ./a.out *.grid
//
/////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SparseCore>

#include <mpi.h>
#include <omp.h>

using Vector = Eigen::VectorXd;
using Vector3 = Eigen::Vector3d;
using SparseMatrix = Eigen::SparseMatrix<double>;

struct Boundary
{
    std::string name_;
    std::string type_;
    int N_;
    std::vector<int> indices_;
    double value_;
};

// Global variables
constexpr double t_min = 0.00;
constexpr double t_max = 100.0;
constexpr double Delta_t = 1.0;
constexpr double rho = 8754.0;
constexpr double C = 380.0;
constexpr double k_diff = 386.0;
constexpr double Q = 40000.0;
constexpr double alpha_heat = k_diff / (rho * C);
constexpr double T_air = 300.0;
constexpr double h = -100.0;

constexpr int numDims = 3;
constexpr int nodesPerFace = 3;
constexpr int nodesPerElement = 4;

constexpr double neumann_source_constant = Q / (3.0 * rho * C);
constexpr double robin_source_constant = T_air * h / (3.0 * rho * C);
constexpr double robin_stiffness_constant = h / (12.0 * rho * C);

const int N_t = static_cast<int>((t_max - t_min) / Delta_t + 1);

int bufferSize = 0;
double* buffer = NULL;

void exchangeData(Vector& T, Boundary* Boundaries, int myN_b)
{
    int yourID = 0;
    int tag = 0;
    MPI_Status status;

    for (int b = 0; b < myN_b; b++)
    {
        if (Boundaries[b].type_ == "interprocess")
        {
            for (int p = 0; p < Boundaries[b].N_; p++)
            {
                buffer[p] = T[Boundaries[b].indices_[p]];
            }
            yourID = static_cast<int>(Boundaries[b].value_);
            MPI_Bsend(buffer, Boundaries[b].N_, MPI_DOUBLE, yourID, tag, MPI_COMM_WORLD);
        }
    }
    for (int b = 0; b < myN_b; b++)
    {
        if (Boundaries[b].type_ == "interprocess")
        {
            yourID = static_cast<int>(Boundaries[b].value_);
            MPI_Recv(buffer, Boundaries[b].N_, MPI_DOUBLE, yourID, tag, MPI_COMM_WORLD, &status);
            for (int p = 0; p < Boundaries[b].N_; p++)
            {
                T[Boundaries[b].indices_[p]] += buffer[p];
            }
        }
    }
}

void readData(char* filename,
              double**& Points,
              int**& Faces,
              int**& Elements,
              Boundary*& Boundaries,
              int& myN_p,
              int& myN_f,
              int& myN_e,
              int& myN_b,
              std::vector<bool>& yourPoints,
              int myID)
{
    std::fstream file;
    std::string temp;
    char myFileName[64];

    int myMaxN_sp = 0;
    int myMaxN_sb = 0;

    int maxN_sp = 0;

    int yourID = 0;

    if (myID == 0)
    {
        std::cout << "Reading " << filename << "'s... " << std::flush;
    }

    sprintf(myFileName, "%s%d", filename, myID);
    file.open(myFileName);

    file >> temp >> myN_p;
    file >> temp >> myN_f;
    file >> temp >> myN_e;
    file >> temp >> myN_b;

    Points = new double*[myN_p];
    Faces = new int*[myN_f];
    Elements = new int*[myN_e];
    Boundaries = new Boundary[myN_b];
    Points[0] = new double[myN_p * numDims];
    Faces[0] = new int[myN_f * nodesPerFace];
    Elements[0] = new int[myN_e * nodesPerElement];

    yourPoints.resize(myN_p, false);

    for (int p = 1, pp = numDims; p < myN_p; p++, pp += numDims)
    {
        Points[p] = &Points[0][pp];
    }
    for (int f = 1, ff = nodesPerFace; f < myN_f; f++, ff += nodesPerFace)
    {
        Faces[f] = &Faces[0][ff];
    }
    for (int e = 1, ee = nodesPerElement; e < myN_e; e++, ee += nodesPerElement)
    {
        Elements[e] = &Elements[0][ee];
    }

    file >> temp;
    for (int p = 0; p < myN_p; p++)
    {
        file >> Points[p][0] >> Points[p][1] >> Points[p][2];
    }

    file >> temp;
    for (int f = 0; f < myN_f; f++)
    {
        file >> Faces[f][0] >> Faces[f][1] >> Faces[f][2];
    }

    file >> temp;
    for (int e = 0; e < myN_e; e++)
    {
        file >> Elements[e][0] >> Elements[e][1] >> Elements[e][2] >> Elements[e][3];
    }

    file >> temp;
    for (int b = 0; b < myN_b; b++)
    {
        file >> Boundaries[b].name_ >> Boundaries[b].type_ >> Boundaries[b].N_;
        Boundaries[b].indices_.resize(Boundaries[b].N_);
        for (int n = 0; n < Boundaries[b].N_; n++)
        {
            file >> Boundaries[b].indices_[n];
        }
        file >> Boundaries[b].value_;
        if (Boundaries[b].type_ == "interprocess")
        {
            myMaxN_sb++;
            myMaxN_sp = std::max(myMaxN_sp, Boundaries[b].N_);
            yourID = static_cast<int>(Boundaries[b].value_);
            if (yourID > myID)
            {
                for (int p = 0; p < Boundaries[b].N_; p++)
                {
                    yourPoints[Boundaries[b].indices_[p]] = true;
                }
            }
        }
    }

    MPI_Allreduce(&myMaxN_sp, &maxN_sp, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    buffer = new double[maxN_sp];
    bufferSize = (maxN_sp * sizeof(double) + MPI_BSEND_OVERHEAD) * myMaxN_sb;
    MPI_Buffer_attach(new char[bufferSize], bufferSize);

    file.close();

    if (myID == 0)
    {
        std::cout << "Done.\n" << std::flush;
    }
}

void writeData(std::fstream& file,
               double* T,
               int myN_p,
               double**& Points,
               int myN_e,
               int**& Elements,
               int l,
               int myID)
{
    char myFileName[64];
    sprintf(myFileName, "Temperature_iter_%03d_%06d.vtk", myID, l);
    file.open(myFileName, std::ios::out);

    // VTK file output
    // Header
    file << "# vtk DataFile Version 3.0"
         << "\n";
    // Title (max 256 char)
    file << "Temperature field of a CPU cooler"
         << "\n";
    // Data Type
    file << "ASCII"
         << "\n";
    // Geometry/topology
    file << "DATASET UNSTRUCTURED_GRID"
         << "\n";
    // Point Coordinates
    file << "POINTS " << myN_p << " double"
         << "\n";
    for (int m = 0; m < myN_p; m++)
    {
        file << Points[m][0] << "\t" << Points[m][1] << "\t" << Points[m][2] << "\n";
    }
    // Element/Cell Nodes
    file << "CELLS " << myN_e << "\t" << 5 * myN_e << "\n";
    for (int e = 0; e < myN_e; e++)
    {
        file << "4"
             << "\t" << Elements[e][0] << "\t" << Elements[e][1] << "\t" << Elements[e][2] << "\t"
             << Elements[e][3] << "\n";
    }
    // Element/Cell types
    file << "CELL_TYPES " << myN_e << "\n";
    for (int e = 0; e < myN_e; e++)
    {
        file << "10"
             << "\n";
    }
    // Temperature
    file << "POINT_DATA " << myN_p << "\n";
    file << "SCALARS "
         << "Temperature "
         << "double"
         << "\n";
    file << "LOOKUP_TABLE "
         << "default"
         << "\n";
    for (int m = 0; m < myN_p; m++)
    {
        file << T[m] << "\n";
    }
    file.close();
}

void assembleSystem(SparseMatrix& M,
                    SparseMatrix& K,
                    Vector& s,
                    Vector& T,
                    double** Points,
                    int** Faces,
                    int** Elements,
                    Boundary* Boundaries,
                    int myN_p,
                    int myN_f,
                    int myN_e,
                    int myN_b,
                    int myID)
{
    if (myID == 0) std::cout << "Assembling system... " << std::flush;

    double x[nodesPerElement];
    double y[nodesPerElement];
    double z[nodesPerElement];

    double M_e[nodesPerElement][nodesPerElement] = {{2.0, 1.0, 1.0, 1.0},
                                                    {1.0, 2.0, 1.0, 1.0},
                                                    {1.0, 1.0, 2.0, 1.0},
                                                    {1.0, 1.0, 1.0, 2.0}};
    double K_e[nodesPerFace][nodesPerFace] = {{2.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 2.0}};

    int Nodes[nodesPerElement] = {0, 0, 0, 0};

    std::vector<double> Omega(myN_e);
    std::vector<double> Gamma(myN_f);

    s.setZero();

    // Calculate face areas
    for (int f = 0; f < myN_f; f++)
    {
        for (int p = 0; p < nodesPerFace; p++)
        {
            x[p] = Points[Faces[f][p]][0];
            y[p] = Points[Faces[f][p]][1];
            z[p] = Points[Faces[f][p]][2];
        }
        Gamma[f] = std::sqrt(
                       std::pow((y[1] - y[0]) * (z[2] - z[0]) - (z[1] - z[0]) * (y[2] - y[0]), 2)
                       + std::pow((z[1] - z[0]) * (x[2] - x[0]) - (x[1] - x[0]) * (z[2] - z[0]), 2)
                       + std::pow((x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0]), 2))
                   / 2.0;
    }

    // Calculate element volumes
    for (int e = 0; e < myN_e; e++)
    {
        for (int p = 0; p < nodesPerElement; p++)
        {
            x[p] = Points[Elements[e][p]][0];
            y[p] = Points[Elements[e][p]][1];
            z[p] = Points[Elements[e][p]][2];
        }

        Omega[e] = std::abs(
            (x[0] * y[1] * z[2] - x[0] * y[2] * z[1] - x[1] * y[0] * z[2] + x[1] * y[2] * z[0]
             + x[2] * y[0] * z[1] - x[2] * y[1] * z[0] - x[0] * y[1] * z[3] + x[0] * y[3] * z[1]
             + x[1] * y[0] * z[3] - x[1] * y[3] * z[0] - x[3] * y[0] * z[1] + x[3] * y[1] * z[0]
             + x[0] * y[2] * z[3] - x[0] * y[3] * z[2] - x[2] * y[0] * z[3] + x[2] * y[3] * z[0]
             + x[3] * y[0] * z[2] - x[3] * y[2] * z[0] - x[1] * y[2] * z[3] + x[1] * y[3] * z[2]
             + x[2] * y[1] * z[3] - x[2] * y[3] * z[1] - x[3] * y[1] * z[2] + x[3] * y[2] * z[1])
            / 6.0);
    }

    // Assemble M, K and s
    std::vector<Eigen::Triplet<double>> K_triplets, M_triplets;

    for (int e = 0; e < myN_e; e++)
    {
        for (int p = 0; p < nodesPerElement; p++)
        {
            Nodes[p] = Elements[e][p];
            x[p] = Points[Nodes[p]][0];
            y[p] = Points[Nodes[p]][1];
            z[p] = Points[Nodes[p]][2];
        }

        double const G[numDims][nodesPerElement] =
            {{(y[3] - y[1]) * (z[2] - z[1]) - (y[2] - y[1]) * (z[3] - z[1]),
              (y[2] - y[0]) * (z[3] - z[2]) - (y[2] - y[3]) * (z[0] - z[2]),
              (y[1] - y[3]) * (z[0] - z[3]) - (y[0] - y[3]) * (z[1] - z[3]),
              (y[0] - y[2]) * (z[1] - z[0]) - (y[0] - y[1]) * (z[2] - z[0])},

             {(x[2] - x[1]) * (z[3] - z[1]) - (x[3] - x[1]) * (z[2] - z[1]),
              (x[3] - x[2]) * (z[2] - z[0]) - (x[0] - x[2]) * (z[2] - z[3]),
              (x[0] - x[3]) * (z[1] - z[3]) - (x[1] - x[3]) * (z[0] - z[3]),
              (x[1] - x[0]) * (z[0] - z[2]) - (x[2] - x[0]) * (z[0] - z[1])},

             {(x[3] - x[1]) * (y[2] - y[1]) - (x[2] - x[1]) * (y[3] - y[1]),
              (x[2] - x[0]) * (y[3] - y[2]) - (x[2] - x[3]) * (y[0] - y[2]),
              (x[1] - x[3]) * (y[0] - y[3]) - (x[0] - x[3]) * (y[1] - y[3]),
              (x[0] - x[2]) * (y[1] - y[0]) - (x[0] - x[1]) * (y[2] - y[0])}};

        // Outer loop over each node
        for (int p = 0; p < nodesPerElement; p++)
        {
            auto const m = Nodes[p];

            Vector3 const Gp(G[0][p], G[1][p], G[2][p]);

            // Inner loop over each node
            for (int q = 0; q < nodesPerElement; q++)
            {
                auto const n = Nodes[q];

                Vector3 const Gq(G[0][q], G[1][q], G[2][q]);

                K_triplets.emplace_back(m, n, -alpha_heat * Gp.dot(Gq) / (36.0 * Omega[e]));
                M_triplets.emplace_back(m, n, M_e[p][q] * Omega[e] / 20.0);
            }
        }
    }

    // Apply boundary conditions
    for (int b = 0; b < myN_b; b++)
    {
        if (Boundaries[b].type_ == "neumann")
        {
            for (int f = 0; f < Boundaries[b].N_; f++)
            {
                for (int p = 0; p < nodesPerFace; p++)
                {
                    Nodes[p] = Faces[Boundaries[b].indices_[f]][p];
                    auto const m = Nodes[p];
                    s[m] += neumann_source_constant * Gamma[Boundaries[b].indices_[f]];
                }
            }
        }
        else if (Boundaries[b].type_ == "robin")
        {
            for (int f = 0; f < Boundaries[b].N_; f++)
            {
                for (int p = 0; p < nodesPerFace; p++)
                {
                    Nodes[p] = Faces[Boundaries[b].indices_[f]][p];
                    auto const m = Nodes[p];
                    s[m] -= robin_source_constant * Gamma[Boundaries[b].indices_[f]];

                    for (int q = 0; q < nodesPerFace; q++)
                    {
                        Nodes[q] = Faces[Boundaries[b].indices_[f]][q];
                        auto const n = Nodes[q];
                        K_triplets.emplace_back(m,
                                                n,
                                                Gamma[Boundaries[b].indices_[f]]
                                                    * robin_stiffness_constant * K_e[p][q]);
                    }
                }
            }
        }
    }

    M.resize(myN_p, myN_p);
    K.resize(myN_p, myN_p);

    K.setFromTriplets(K_triplets.begin(), K_triplets.end());
    M.setFromTriplets(M_triplets.begin(), M_triplets.end());

    MPI_Barrier(MPI_COMM_WORLD);

    if (myID == 0) std::cout << "Done.\n" << std::flush;
}

double computeInnerProduct(Vector const& v1,
                           Vector const& v2,
                           std::vector<bool> const& yourPoints,
                           int N_row)
{
    double myInnerProduct{0.0};
    double innerProduct{0.0};

    for (int m = 0; m < N_row; m++)
    {
        if (!yourPoints[m])
        {
            myInnerProduct += v1[m] * v2[m];
        }
    }
    MPI_Allreduce(&myInnerProduct, &innerProduct, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return innerProduct;
}

void linear_solve(SparseMatrix& A,
                  Vector& u,
                  Vector& b,
                  Boundary* Boundaries,
                  std::vector<bool> const& yourPoints,
                  int myN_b,
                  int myID)
{
    auto const N_row = A.rows();

    if (myID == 0) std::cout << "Solving..." << std::endl;

    // Compute the initial residual
    Vector Au = A * u;

    exchangeData(Au, Boundaries, myN_b);

    Vector r_old = b - Au;

    Vector d = r_old;

    auto r_oldTr_old = computeInnerProduct(r_old, r_old, yourPoints, N_row);

    auto const first_r_norm = std::sqrt(r_oldTr_old);
    auto r_norm = first_r_norm;

    // Conjugate Gradient iterative loop
    double tolerance = 1.0e-8;
    int maxIterations = 2000, iter = 0;
    while (r_norm > tolerance && iter < maxIterations)
    {
        Vector Ad = A * d;

        exchangeData(Ad, Boundaries, myN_b);

        auto const dTAd = computeInnerProduct(d, Ad, yourPoints, N_row);

        auto const alpha = r_oldTr_old / dTAd;

        u += alpha * d;

        Vector const r = r_old - alpha * Ad;

        auto const rTr = computeInnerProduct(r, r, yourPoints, N_row);

        auto const beta = rTr / r_oldTr_old;

        d = r + beta * d;
        r_old = r;

        r_oldTr_old = rTr;

        r_norm = std::sqrt(rTr) / first_r_norm;

        iter++;
    }

    if (myID == 0) std::cout << ", iter = " << iter << ", r_norm = " << r_norm << std::endl;
}

int main(int argc, char** argv)
{
    // Memory Allocation
    double** Points = NULL;
    int** Faces = NULL;
    int** Elements = NULL;
    Boundary* Boundaries = NULL;

    std::vector<bool> yourPoints;

    double* buffer = NULL;
    int myN_p = 0;
    int myN_f = 0;
    int myN_e = 0;
    int myN_b = 0;

    int myID = 0;
    int N_Processes = 0;

    double t = 0.0;

    std::fstream file;

    double* Tmax = new double[N_t];
    double tempTmax;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myID);
    MPI_Comm_size(MPI_COMM_WORLD, &N_Processes);

    if (argc < 2)
    {
        if (myID == 0) std::cerr << "No grid file specified" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    readData(argv[1], Points, Faces, Elements, Boundaries, myN_p, myN_f, myN_e, myN_b, yourPoints, myID);

    double wtime = (myID == 0) ? MPI_Wtime() : 0.0;

    // Set initial condition
    t = t_min;
    Vector u = Vector::Ones(myN_p) * T_air;

    // Allocate arrays
    Vector s(myN_p), b(myN_p);
    SparseMatrix M, K;

    assembleSystem(M, K, s, u, Points, Faces, Elements, Boundaries, myN_p, myN_f, myN_e, myN_b, myID);

    SparseMatrix A = M - Delta_t * K;

    exchangeData(s, Boundaries, myN_b);

    // writeData(file, T, myN_p, Points, myN_e, Elements,0,myID);

    // Get Max Temperature
    tempTmax = u.maxCoeff();
    MPI_Allreduce(&tempTmax, &Tmax[0], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Time marching loop
    for (int l = 1; l < N_t; l++)
    {
        t += Delta_t;

        if (myID == 0) std::cout << "t = " << t << std::endl;

        b = M * u;

        exchangeData(b, Boundaries, myN_b);

        b += Delta_t * s;

        linear_solve(A, u, b, Boundaries, yourPoints, myN_b, myID);

        // Get Max Temperature
        tempTmax = u.maxCoeff();
        MPI_Allreduce(&tempTmax, &Tmax[l], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    if (myID == 0)
    {
        wtime = MPI_Wtime() - wtime; // Record the end time and calculate elapsed time
        std::cout << "Simulation took " << wtime << " seconds with " << N_Processes << " processes"
                  << std::endl;

        // Write maximum temperature to file
        file.open("maxTemperature.data", std::ios::out);
        for (int m = 0; m < N_t; m++)
        {
            file << Tmax[m] << "\n";
        }
        file.close();
    }

    MPI_Buffer_detach(&buffer, &bufferSize);

    // Deallocate arrays
    delete[] Points[0];
    delete[] Points;
    delete[] Faces[0];
    delete[] Faces;
    delete[] Elements[0];
    delete[] Elements;
    delete[] Boundaries;

    delete[] buffer;

    MPI_Finalize();

    return 0;
}
