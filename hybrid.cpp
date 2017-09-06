/////////////////////////////////////////////////////////////////////////////
//
//
// Problem:		dT/dt = alpha_heat * Grad^2 T
//
// Method:		Finite Element Method with linear 3D tetrahedral elements
//				and Implicit Euler Method and Conjugate Gradient Method
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

struct Boundary
{
    std::string name_;
    std::string type_;
    int N_;
    std::vector<int> indices_;
    int neighbour_process_;
};

using Vector = Eigen::VectorXd;
using Vector3 = Eigen::Vector3d;
using SparseMatrix = Eigen::SparseMatrix<double>;

using Points = std::vector<std::array<double, 3>>;
using Faces = std::vector<std::array<int, 3>>;
using Elements = std::vector<std::array<int, 4>>;
using Boundaries = std::vector<Boundary>;

class ParallelCommunicator
{
public:
    ParallelCommunicator(int argc, char** argv)
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &processes);

        start_time = MPI_Wtime();
    }

    ~ParallelCommunicator()
    {
        if (buffer_send_size > 0)
        {
            MPI_Buffer_detach(&buffer_send, &buffer_send_size);
            delete[] buffer_send;
            buffer_send = nullptr;
        }
        MPI_Finalize();
    }

    auto elapsed_time() const { return MPI_Wtime() - start_time; }

    auto process_rank() const { return rank; }

    auto is_master() const { return rank == 0; }

    auto number_of_processes() const { return processes; }

    void abort() const { MPI_Abort(MPI_COMM_WORLD, 1); }

    void barrier() const { MPI_Barrier(MPI_COMM_WORLD); }

    // void wait() const {}

    // void wait_all() const {}

    double maximum(double const partial_max) const
    {
        double total_max;
        MPI_Allreduce(&partial_max, &total_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return total_max;
    }

    int maximum(int const partial_max) const
    {
        int total_max;
        MPI_Allreduce(&partial_max, &total_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        return total_max;
    }

    double sum(double const partial_sum) const
    {
        double total_sum;
        MPI_Allreduce(&partial_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return total_sum;
    }

    int sum(int const partial_sum) const
    {
        int total_sum;
        MPI_Allreduce(&partial_sum, &total_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        return total_sum;
    }

    std::vector<int> receive(std::vector<int> buffer, int const source) const
    {
        MPI_Status status;

        // Probe for an incoming message from process zero
        MPI_Probe(source, 0, MPI_COMM_WORLD, &status);

        // When probe returns, the status object has the size and other
        // attributes of the incoming message. Get the message size
        int buffer_size = 0;
        MPI_Get_count(&status, MPI_INT, &buffer_size);

        // Allocate a buffer to hold the incoming numbers
        buffer.resize(buffer_size);

        // Now receive the message with the allocated buffer
        MPI_Recv(buffer.data(), buffer_size, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        return buffer;
    }

    std::vector<double> receive(std::vector<double> buffer, int const source) const
    {
        MPI_Status status;

        // Probe for an incoming message
        MPI_Probe(source, 0, MPI_COMM_WORLD, &status);

        // When probe returns, the status object has the size and other
        // attributes of the incoming message. Get the message size
        int buffer_size = 0;
        MPI_Get_count(&status, MPI_DOUBLE, &buffer_size);

        // Allocate a buffer to hold the incoming numbers
        buffer.resize(buffer_size);

        // Now receive the message with the allocated buffer
        MPI_Recv(buffer.data(), buffer.size(), MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        return buffer;
    }

    void allocate_double_buffer(int const inner_size, int const outer_size)
    {
        // Include the buffer overhead in the computation
        buffer_send_size = (inner_size * sizeof(double) + MPI_BSEND_OVERHEAD) * outer_size;

        MPI_Buffer_attach(new char[buffer_send_size], buffer_send_size);
    }

    void buffered_send(std::vector<double> const& buffer, int const destination) const
    {
        assert(buffer_send_size >= buffer.size());

        MPI_Bsend(buffer.data(),   // Initial buffer address
                  buffer.size(),   // Size
                  MPI_DOUBLE,      // Datatype
                  destination,     // Destination
                  0,               // Message tag
                  MPI_COMM_WORLD); // Communicator
    }

private:
    int rank = 0;
    int processes = 1;

    double start_time = 0.0;

    char* buffer_send = nullptr;
    int buffer_send_size = 0;
};

void exchange_data(Vector& u, Boundaries const& boundaries, ParallelCommunicator const& parallel)
{
    for (auto const& boundary : boundaries)
    {
        if (boundary.type_ == "interprocess")
        {
            std::vector<double> buffer;
            buffer.reserve(boundary.N_);

            for (auto const& i : boundary.indices_) buffer.push_back(u[i]);

            parallel.buffered_send(buffer, boundary.neighbour_process_);
        }
    }

    for (auto const& boundary : boundaries)
    {
        if (boundary.type_ == "interprocess")
        {
            auto const buffer = parallel.receive(std::vector<double>{}, boundary.neighbour_process_);

            for (int p = 0; p < boundary.N_; p++)
            {
                u[boundary.indices_[p]] += buffer[p];
            }
        }
    }
}

void read_data(std::string const& filename,
               Points& points,
               Faces& faces,
               Elements& elements,
               Boundaries& boundaries,
               std::vector<bool>& yourPoints,
               ParallelCommunicator& parallel)
{
    std::fstream file;
    std::string temp;

    if (parallel.is_master()) std::cout << "Reading " << filename << "'s... " << std::flush;

    file.open(filename + std::to_string(parallel.process_rank()));

    int myN_p, myN_f, myN_e, myN_b;
    file >> temp >> myN_p >> temp >> myN_f >> temp >> myN_e >> temp >> myN_b;

    points.resize(myN_p);
    faces.resize(myN_f);
    elements.resize(myN_e);
    yourPoints.resize(myN_p, false);
    boundaries.resize(myN_b);

    file >> temp;
    for (auto& point : points) file >> point[0] >> point[1] >> point[2];

    file >> temp;
    for (auto& face : faces) file >> face[0] >> face[1] >> face[2];

    file >> temp;
    for (auto& element : elements) file >> element[0] >> element[1] >> element[2] >> element[3];

    int max_local_shared_points = 0;
    int local_shared_boundaries = 0;

    file >> temp;
    for (auto& boundary : boundaries)
    {
        file >> boundary.name_ >> boundary.type_ >> boundary.N_;

        boundary.indices_.resize(boundary.N_);

        for (auto& index : boundary.indices_) file >> index;

        double neighbour_process{0.0};

        file >> neighbour_process;

        boundary.neighbour_process_ = static_cast<int>(neighbour_process);

        if (boundary.type_ == "interprocess")
        {
            local_shared_boundaries++;

            max_local_shared_points = std::max(max_local_shared_points, boundary.N_);

            if (boundary.neighbour_process_ > parallel.process_rank())
            {
                for (auto const& index : boundary.indices_) yourPoints[index] = true;
            }
        }
    }

    std::cout << "Process " << parallel.process_rank() << " has " << max_local_shared_points
              << " shared points and " << local_shared_boundaries << " shared boundaries\n";

    // Determine the maximum size the buffer needs to be
    int const shared_points = parallel.maximum(max_local_shared_points);

    parallel.allocate_double_buffer(shared_points, local_shared_boundaries);

    // Allocate the largest buffer required
    // buffer = new double[shared_points];

    // Include the buffer overhead in the computation
    // bufferSize = (shared_points * sizeof(double) + MPI_BSEND_OVERHEAD) * local_shared_boundaries;

    // MPI_Buffer_attach(new char[bufferSize], bufferSize);

    file.close();

    if (parallel.is_master()) std::cout << "Done.\n" << std::flush;
}

void write_data(Vector const& T,
                Points const& points,
                Elements const& elements,
                int l,
                ParallelCommunicator const& parallel)
{
    char myFileName[64];
    sprintf(myFileName, "Temperature_iter_%03d_%06d.vtk", parallel.process_rank(), l);

    std::fstream file;
    file.open(myFileName, std::ios::out);

    auto const myN_p = points.size();
    auto const myN_e = elements.size();

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
    for (auto const& point : points)
    {
        file << point[0] << "\t" << point[1] << "\t" << point[2] << "\n";
    }
    // Element/Cell Nodes
    file << "CELLS " << myN_e << "\t" << 5 * myN_e << "\n";
    for (auto const& element : elements)
    {
        file << "4"
             << "\t" << element[0] << "\t" << element[1] << "\t" << element[2] << "\t" << element[3]
             << "\n";
    }
    // Element/Cell types
    file << "CELL_TYPES " << myN_e << "\n";
    for (auto const& element : elements)
    {
        file << "10\n";
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
    for (int m = 0; m < T.size(); m++)
    {
        file << T[m] << "\n";
    }
    file.close();
}

void assemble_system(SparseMatrix& M,
                     SparseMatrix& K,
                     Vector& s,
                     Vector& T,
                     Points const& points,
                     Faces const& faces,
                     Elements const& elements,
                     Boundaries& boundaries,
                     ParallelCommunicator const& parallel)
{
    if (parallel.is_master()) std::cout << "Assembling system... " << std::flush;

    double x[nodesPerElement];
    double y[nodesPerElement];
    double z[nodesPerElement];

    double M_e[nodesPerElement][nodesPerElement] = {{2.0, 1.0, 1.0, 1.0},
                                                    {1.0, 2.0, 1.0, 1.0},
                                                    {1.0, 1.0, 2.0, 1.0},
                                                    {1.0, 1.0, 1.0, 2.0}};
    double K_e[nodesPerFace][nodesPerFace] = {{2.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 2.0}};

    int Nodes[nodesPerElement] = {0, 0, 0, 0};

    std::vector<double> Omega(elements.size());
    std::vector<double> Gamma(faces.size());

    s.setZero();

    // Calculate face areas
    for (int f = 0; f < faces.size(); f++)
    {
        for (int p = 0; p < nodesPerFace; p++)
        {
            x[p] = points[faces[f][p]][0];
            y[p] = points[faces[f][p]][1];
            z[p] = points[faces[f][p]][2];
        }
        Gamma[f] = std::sqrt(
                       std::pow((y[1] - y[0]) * (z[2] - z[0]) - (z[1] - z[0]) * (y[2] - y[0]), 2)
                       + std::pow((z[1] - z[0]) * (x[2] - x[0]) - (x[1] - x[0]) * (z[2] - z[0]), 2)
                       + std::pow((x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0]), 2))
                   / 2.0;
    }

    // Calculate element volumes
    for (int e = 0; e < elements.size(); e++)
    {
        for (int p = 0; p < nodesPerElement; p++)
        {
            x[p] = points[elements[e][p]][0];
            y[p] = points[elements[e][p]][1];
            z[p] = points[elements[e][p]][2];
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

    for (int e = 0; e < elements.size(); e++)
    {
        for (int p = 0; p < nodesPerElement; p++)
        {
            Nodes[p] = elements[e][p];
            x[p] = points[Nodes[p]][0];
            y[p] = points[Nodes[p]][1];
            z[p] = points[Nodes[p]][2];
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
    for (int b = 0; b < boundaries.size(); b++)
    {
        if (boundaries[b].type_ == "neumann")
        {
            for (int f = 0; f < boundaries[b].N_; f++)
            {
                for (int p = 0; p < nodesPerFace; p++)
                {
                    Nodes[p] = faces[boundaries[b].indices_[f]][p];
                    auto const m = Nodes[p];
                    s[m] += neumann_source_constant * Gamma[boundaries[b].indices_[f]];
                }
            }
        }
        else if (boundaries[b].type_ == "robin")
        {
            for (int f = 0; f < boundaries[b].N_; f++)
            {
                for (int p = 0; p < nodesPerFace; p++)
                {
                    Nodes[p] = faces[boundaries[b].indices_[f]][p];
                    auto const m = Nodes[p];
                    s[m] -= robin_source_constant * Gamma[boundaries[b].indices_[f]];

                    for (int q = 0; q < nodesPerFace; q++)
                    {
                        Nodes[q] = faces[boundaries[b].indices_[f]][q];
                        auto const n = Nodes[q];
                        K_triplets.emplace_back(m,
                                                n,
                                                Gamma[boundaries[b].indices_[f]]
                                                    * robin_stiffness_constant * K_e[p][q]);
                    }
                }
            }
        }
    }

    M.resize(points.size(), points.size());
    K.resize(points.size(), points.size());

    K.setFromTriplets(K_triplets.begin(), K_triplets.end());
    M.setFromTriplets(M_triplets.begin(), M_triplets.end());

    parallel.barrier();
    if (parallel.is_master()) std::cout << "done" << std::endl;
    parallel.barrier();
}

double compute_dot_product(Vector const& v1,
                           Vector const& v2,
                           std::vector<bool> const& yourPoints,
                           ParallelCommunicator const& parallel)
{
    double inner_product{0.0};

    assert(v1.rows() == v2.rows());

    for (auto m = 0; m < v1.rows(); m++)
    {
        if (!yourPoints[m])
        {
            inner_product += v1[m] * v2[m];
        }
    }
    return parallel.sum(inner_product);
}

void linear_solve(SparseMatrix const& A,
                  Vector& u,
                  Vector& b,
                  Boundaries& boundaries,
                  std::vector<bool> const& yourPoints,
                  ParallelCommunicator const& parallel)
{
    // Compute the initial residual
    Vector Au = A * u;

    exchange_data(Au, boundaries, parallel);

    Vector r_old = b - Au;

    Vector d = r_old;

    auto r_oldTr_old = compute_dot_product(r_old, r_old, yourPoints, parallel);

    auto const first_r_norm = std::sqrt(r_oldTr_old);
    auto r_norm = first_r_norm;

    // Conjugate Gradient iterative loop
    double tolerance = 1.0e-8;
    int maxIterations = 2000, iter = 0;
    while (r_norm > tolerance && iter < maxIterations)
    {
        Vector Ad = A * d;

        exchange_data(Ad, boundaries, parallel);

        auto const dTAd = compute_dot_product(d, Ad, yourPoints, parallel);

        auto const alpha = r_oldTr_old / dTAd;

        u += alpha * d;

        Vector const r = r_old - alpha * Ad;

        auto const rTr = compute_dot_product(r, r, yourPoints, parallel);

        auto const beta = rTr / r_oldTr_old;

        d = r + beta * d;
        r_old = r;

        r_oldTr_old = rTr;

        r_norm = std::sqrt(rTr) / first_r_norm;

        iter++;
    }

    if (parallel.is_master())
    {
        std::cout << "Conjugate gradient iterations = " << iter << ", residual = " << r_norm
                  << std::endl;
    }
}

int main(int argc, char** argv)
{
    ParallelCommunicator parallel(argc, argv);

    Points points;
    Faces faces;
    Elements elements;
    Boundaries boundaries;
    std::vector<bool> yourPoints;
    std::vector<double> Tmax(N_t);

    if (argc < 2)
    {
        if (parallel.is_master()) std::cerr << "No grid file specified" << std::endl;
        parallel.abort();
    }

    read_data(argv[1], points, faces, elements, boundaries, yourPoints, parallel);

    // Set initial condition
    double t = t_min;
    Vector u = Vector::Ones(points.size()) * T_air;

    // Allocate arrays
    Vector s(points.size()), b(points.size());

    SparseMatrix M, K;

    assemble_system(M, K, s, u, points, faces, elements, boundaries, parallel);

    SparseMatrix const A = M - Delta_t * K;

    exchange_data(s, boundaries, parallel);

    write_data(u, points, elements, 0, parallel);

    Tmax[0] = parallel.maximum(u.maxCoeff());

    // Time solver loop
    for (int l = 1; l < N_t; l++)
    {
        t += Delta_t;

        if (parallel.is_master()) std::cout << "t = " << t << std::endl;

        b = M * u;

        exchange_data(b, boundaries, parallel);

        b += Delta_t * s;

        linear_solve(A, u, b, boundaries, yourPoints, parallel);

        Tmax[l] = parallel.maximum(u.maxCoeff());
    }

    if (parallel.is_master())
    {
        std::cout << "Simulation took " << parallel.elapsed_time() << " seconds with "
                  << parallel.number_of_processes() << " processes" << std::endl;

        // Write maximum temperature to file
        std::fstream file;
        file.open("maxTemperature.data", std::ios::out);
        for (auto const T : Tmax) file << T << std::endl;
        file.close();
    }
    return 0;
}
