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
#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>

using namespace std;

// Class definitions
class SparseMatrix
{
public:
    SparseMatrix(int nrow, int nnzperrow)
    {
        // This constructor is called if we happen to know the number of rows
        // and an estimate of the number of nonzero entries per row.
        this->initialize(nrow, nnzperrow);
    }

    SparseMatrix()
    {
        // This constructor is called if we have no useful information
        N_row_       = 0;
        N_nz_        = 0;
        N_nz_rowmax_ = 0;
        allocSize_   = 0;
        val_         = NULL;
        col_         = NULL;
        row_         = NULL;
        nnzs_        = NULL;
    }

    ~SparseMatrix()
    {
        if (val_) delete[] val_;
        if (col_) delete[] col_;
        if (row_) delete[] row_;
        if (nnzs_) delete[] nnzs_;
    }

    void initialize(int nrow, int nnzperrow)
    {
        N_row_       = nrow;
        N_nz_        = 0;
        N_nz_rowmax_ = nnzperrow;
        allocSize_   = N_row_ * N_nz_rowmax_;
        val_         = new double[allocSize_];
        col_         = new int[allocSize_];
        row_         = new int[N_row_ + 1];
        nnzs_        = new int[N_row_ + 1];

        memset(val_, 0, allocSize_ * sizeof(double));
        memset(col_, -1, allocSize_ * sizeof(int));
        memset(row_, 0, (N_row_ + 1) * sizeof(int));
        memset(nnzs_, 0, (N_row_ + 1) * sizeof(int));

        for (int k = 0, kk = 0; k < N_row_; k++, kk += N_nz_rowmax_)
        {
            row_[k] = kk;
        }
        return;
    }

    void finalize()
    {
        int minCol    = 0;
        int insertPos = 0;
        int index     = 0;

        // Now that the matrix is assembled we can set N_nz_rowmax_ explicitly by
        // taking the largest value in the nnzs_ array
        N_nz_rowmax_ = 0;
        for (int m = 0; m < N_row_; m++)
        {
            N_nz_rowmax_ = max(N_nz_rowmax_, nnzs_[m]);
        }

        double* tempVal = new double[N_nz_];
        int* tempCol    = new int[N_nz_];
        int* tempRow    = new int[N_row_ + 1];
        // This array will help us sort the column indices
        bool* isSorted = new bool[allocSize_];

        memset(tempVal, 0, N_nz_ * sizeof(double));
        memset(tempCol, 0, N_nz_ * sizeof(int));
        memset(tempRow, 0, (N_row_ + 1) * sizeof(int));
        memset(isSorted, 0, allocSize_ * sizeof(bool));

        for (int m = 0; m < N_row_; m++)
        {
            for (int k = row_[m]; k < (row_[m] + nnzs_[m]); k++)
            {
                minCol = N_row_ + 1;
                for (int kk = row_[m]; kk < (row_[m] + nnzs_[m]); kk++)
                {
                    if (!isSorted[kk] && col_[kk] < minCol)
                    {
                        index  = kk;
                        minCol = col_[index];
                    }
                }
                tempVal[insertPos] = val_[index];
                tempCol[insertPos] = col_[index];
                isSorted[index]    = true;
                insertPos++;
            }
            tempRow[m + 1] = tempRow[m] + nnzs_[m];
        }

        delete[] val_;
        delete[] col_;
        delete[] row_;
        delete[] nnzs_;
        delete[] isSorted;

        val_       = tempVal;
        col_       = tempCol;
        row_       = tempRow;
        nnzs_      = NULL;
        allocSize_ = N_nz_;

        return;
    }

    inline double& operator()(int m, int n)
    {

        // If the arrays are already full and inserting this entry would cause us to
        // run off the end,
        // then we'll need to resize the arrays before inserting it
        if (nnzs_[m] >= N_nz_rowmax_)
        {
            this->reallocate();
        }

        int k           = row_[m];
        bool foundEntry = false;

        // Search between row(m) and row(m+1) for col(k) = n (i.e. is the entry
        // already in the matrix)
        while (k < (row_[m] + nnzs_[m]) && !foundEntry)
        {
            if (col_[k] == n)
            {
                foundEntry = true;
            }
            k++;
        }
        // If the entry is already in the matrix, then return a reference to it
        if (foundEntry)
        {
            return val_[k - 1];
        }
        // If the entry is not already in the matrix then we'll need to insert it
        else
        {
            N_nz_++;
            nnzs_[m]++;
            col_[k] = n;
            return val_[k];
        }
    }

    inline double& operator()(int k) { return val_[k]; }

    void operator=(const SparseMatrix& A)
    {
        if (val_) delete[] val_;
        if (col_) delete[] col_;
        if (row_) delete[] row_;
        if (nnzs_) delete[] nnzs_;

        N_row_       = A.N_row_;
        N_nz_        = A.N_nz_;
        N_nz_rowmax_ = A.N_nz_rowmax_;
        allocSize_   = A.allocSize_;
        val_         = new double[allocSize_];
        col_         = new int[allocSize_];
        row_         = new int[N_row_ + 1];

        memcpy(val_, A.val_, N_nz_ * sizeof(double));
        memcpy(col_, A.col_, N_nz_ * sizeof(int));
        memcpy(row_, A.row_, (N_row_ + 1) * sizeof(int));
    }
    inline void multiply(double* u, double* v)
    {
// Note: This function will perform a matrix vector multiplication with the
// input vector v, returning the output in u.
#pragma omp for
        for (int m = 0; m < N_row_; m++)
        {
            u[m] = 0.0;
            for (int k = row_[m]; k < row_[m + 1]; k++)
            {
                u[m] += val_[k] * v[col_[k]];
            }
        }
        return;
    }
    inline void subtract(double u, SparseMatrix& A)
    {
#pragma omp for
        for (int k = 0; k < N_nz_; k++)
        {
            val_[k] -= (u * A.val_[k]);
        }

        return;
    }
    inline int getNnz() { return N_nz_; }
    inline int getNrow() { return N_row_; }
    void print(const char* name)
    {
        fstream matrix;
        cout << "Matrix " << name << " has " << N_row_ << " rows with " << N_nz_
             << " non-zero entries - " << allocSize_ << " allocated." << flush;
        matrix.open(name, ios::out);
        matrix << "Mat = [" << endl;
        for (int m = 0; m < N_row_; m++)
        {
            for (int n = row_[m]; n < row_[m + 1]; n++)
            {
                matrix << m + 1 << "\t" << col_[n] + 1 << "\t" << val_[n] << endl;
            }
        }
        matrix << "];" << endl;
        matrix.close();
        cout << " Done." << flush << endl;
        return;
    }

protected:
    void reallocate()
    {
        // Double the memory allocation size
        N_nz_rowmax_ *= 2;

        allocSize_ = N_nz_rowmax_ * N_row_;

        // Create some temporary arrays of the new size
        double* tempVal = new double[allocSize_];
        int* tempCol    = new int[allocSize_];

        memset(tempVal, 0, allocSize_ * sizeof(double));
        memset(tempCol, 0, allocSize_ * sizeof(int));

        for (int m = 0, mm = 0; m < N_row_; m++, mm += N_nz_rowmax_)
        {
            memcpy(&tempVal[mm], &val_[row_[m]], nnzs_[m] * sizeof(double));
            memcpy(&tempCol[mm], &col_[row_[m]], nnzs_[m] * sizeof(int));
            row_[m] = mm;
        }

        // Delete the memory used by the old arrays
        delete[] val_;
        delete[] col_;

        // Assign the addresses of the new arrays
        val_ = tempVal;
        col_ = tempCol;

        return;
    }

private:
    double* val_;     // [N_nz]    Stores the nonzero elements of the matrix
    int* col_;        // [N_nz]    Stores the column indices of the elements in each row
    int* row_;        // [N_row+1] Stores the locations in val that start a row
    int* nnzs_;       // [N_row+1] Stores the number of nonzero entries per row during
                      // the assembly process
    int N_row_;       // The number of rows in the matrix
    int N_nz_;        // The number of non-zero entries currently stored in the matrix
    int N_nz_rowmax_; // The maximum number of non-zero entries per row. This will
                      // be an estimate until the matrix is assembled
    int allocSize_;   // The number of non-zero entries currently allocated for in
                      // val_ and col_
};

class Boundary
{
public:
    Boundary() {}
    string name_;
    string type_;
    int N_;
    int* indices_;
    double value_;
};

// Global variables
const double t_min        = 0.00;
const double t_max        = 100.0;
const double Delta_t      = 1.0;
const double rho          = 8754.0;
const double C            = 380.0;
const double k_diff       = 386.0;
const double Q            = 40000.0;
const double alpha_heat   = k_diff / (rho * C);
const double T_air        = 300.0;
const double h            = -100.0;
const int numDims         = 3;
const int nodesPerFace    = 3;
const int nodesPerElement = 4;

const double neumann_source_constant  = Q / (3.0 * rho * C);
const double robin_source_constant    = T_air * h / (3.0 * rho * C);
const double robin_stiffness_constant = h / (12.0 * rho * C);

const int N_t  = static_cast<int>((t_max - t_min) / Delta_t + 1);
int bufferSize = 0;
double* buffer = NULL;

// Function declarations
void readData(char* filename,
              double**& Points,
              int**& Faces,
              int**& Elements,
              Boundary*& Boundaries,
              int& myN_p,
              int& myN_f,
              int& myN_e,
              int& myN_b,
              bool*& yourPoints,
              int myID);
void writeData(fstream& file,
               double* T,
               int myN_p,
               double**& Points,
               int myN_e,
               int**& Elements,
               int l,
               int myID);
void assembleSystem(SparseMatrix& M,
                    SparseMatrix& K,
                    double* s,
                    double* T,
                    double** Points,
                    int** Faces,
                    int** Elements,
                    Boundary* Boundaries,
                    int myN_p,
                    int myN_f,
                    int myN_e,
                    int myN_b,
                    int myID);
void solve(SparseMatrix& A,
           double* T,
           double* b,
           Boundary* Boundaries,
           bool* yourPoints,
           int myN_b,
           int myID);
void exchangeData(double* T, Boundary* Boundaries, int myN_b);
double computeInnerProduct(double* v1, double* v2, bool* yourPoints, int N_row);

int main(int argc, char** argv)
{
    // Memory Allocation
    double** Points      = NULL;
    int** Faces          = NULL;
    int** Elements       = NULL;
    Boundary* Boundaries = NULL;
    bool* yourPoints     = NULL;
    double* buffer       = NULL;
    int myN_p            = 0;
    int myN_f            = 0;
    int myN_e            = 0;
    int myN_b            = 0;
    int myID             = 0;
    int N_Processes      = 0;
    double t             = 0.0;
    fstream file;
    double wtime;
    double* Tmax = new double[N_t];
    double tempTmax;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myID);
    MPI_Comm_size(MPI_COMM_WORLD, &N_Processes);

    if (argc < 2)
    {
        if (myID == 0)
        {
            cerr << "No grid file specified" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    readData(argv[1],
             Points,
             Faces,
             Elements,
             Boundaries,
             myN_p,
             myN_f,
             myN_e,
             myN_b,
             yourPoints,
             myID);

    // Allocate arrays
    double* T = new double[myN_p];
    double* s = new double[myN_p];
    double* b = new double[myN_p];
    SparseMatrix M;
    SparseMatrix K;
    SparseMatrix A;

    if (myID == 0)
    {
        wtime = MPI_Wtime();
    }

    // Set initial condition
    t = t_min;
    for (int m = 0; m < myN_p; m++)
    {
        T[m] = T_air;
    }

    assembleSystem(M,
                   K,
                   s,
                   T,
                   Points,
                   Faces,
                   Elements,
                   Boundaries,
                   myN_p,
                   myN_f,
                   myN_e,
                   myN_b,
                   myID);

    A = M;

    A.subtract(Delta_t, K); // At this point we have A = M-Delta_t*K

    exchangeData(s, Boundaries, myN_b);

    // writeData(file, T, myN_p, Points, myN_e, Elements,0,myID);

    // Get Max Temperature
    tempTmax = *std::max_element(T, T + myN_p);
    MPI_Allreduce(&tempTmax, &Tmax[0], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Time marching loop
    for (int l = 1; l < N_t; l++)
    {
        t += Delta_t;
        if (myID == 0)
        {
            cout << "t = " << t << endl;
        }

        // Assemble b
        M.multiply(b, T);

        exchangeData(b, Boundaries, myN_b);

        for (int m = 0; m < myN_p; m++)
        {
            b[m] += Delta_t * s[m];
        }

        // Solve the linear system
        solve(A, T, b, Boundaries, yourPoints, myN_b, myID);

        // Get Max Temperature
        tempTmax = *std::max_element(T, T + myN_p);
        MPI_Allreduce(&tempTmax, &Tmax[l], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    } // end of time loop

    if (myID == 0)
    {
        wtime = MPI_Wtime() - wtime; // Record the end time and calculate elapsed time
        cout << "Simulation took " << wtime << " seconds with " << N_Processes
             << " processes" << endl;

        // Write maximum temperature to file
        file.open("maxTemperature.data", ios::out);
        for (int m = 0; m < N_t; m++)
        {
            file << Tmax[m] << "\n";
        }
        file.close();
    }

    MPI_Buffer_detach(&buffer, &bufferSize);

    // Deallocate arrays
    for (int boundary = 0; boundary < myN_b; boundary++)
    {
        delete[] Boundaries[boundary].indices_;
    }
    delete[] Points[0];
    delete[] Points;
    delete[] Faces[0];
    delete[] Faces;
    delete[] Elements[0];
    delete[] Elements;
    delete[] Boundaries;
    delete[] T;
    delete[] s;
    delete[] b;
    delete[] buffer;

    MPI_Finalize();

    return 0;
}

void exchangeData(double* T, Boundary* Boundaries, int myN_b)
{
    int yourID = 0;
    int tag    = 0;
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
            MPI_Recv(buffer,
                     Boundaries[b].N_,
                     MPI_DOUBLE,
                     yourID,
                     tag,
                     MPI_COMM_WORLD,
                     &status);
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
              bool*& yourPoints,
              int myID)
{
    fstream file;
    string temp;
    char myFileName[64];
    int myMaxN_sp = 0;
    int myMaxN_sb = 0;
    int maxN_sp   = 0;
    int yourID    = 0;

    if (myID == 0)
    {
        cout << "Reading " << filename << "'s... " << flush;
    }

    sprintf(myFileName, "%s%d", filename, myID);
    file.open(myFileName);

    file >> temp >> myN_p;
    file >> temp >> myN_f;
    file >> temp >> myN_e;
    file >> temp >> myN_b;

    Points      = new double*[myN_p];
    Faces       = new int*[myN_f];
    Elements    = new int*[myN_e];
    Boundaries  = new Boundary[myN_b];
    Points[0]   = new double[myN_p * numDims];
    Faces[0]    = new int[myN_f * nodesPerFace];
    Elements[0] = new int[myN_e * nodesPerElement];
    yourPoints  = new bool[myN_p];

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

    // sets all yourPoints to false
    memset(yourPoints, false, myN_p * sizeof(bool));

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
        Boundaries[b].indices_ = new int[Boundaries[b].N_];
        for (int n = 0; n < Boundaries[b].N_; n++)
        {
            file >> Boundaries[b].indices_[n];
        }
        file >> Boundaries[b].value_;
        if (Boundaries[b].type_ == "interprocess")
        {
            myMaxN_sb++;
            myMaxN_sp = std::max(myMaxN_sp, Boundaries[b].N_);
            yourID    = static_cast<int>(Boundaries[b].value_);
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
    buffer     = new double[maxN_sp];
    bufferSize = (maxN_sp * sizeof(double) + MPI_BSEND_OVERHEAD) * myMaxN_sb;
    MPI_Buffer_attach(new char[bufferSize], bufferSize);

    file.close();

    if (myID == 0)
    {
        cout << "Done.\n" << flush;
    }

    return;
} // end of readData

void writeData(fstream& file,
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
    file.open(myFileName, ios::out);

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
             << "\t" << Elements[e][0] << "\t" << Elements[e][1] << "\t" << Elements[e][2]
             << "\t" << Elements[e][3] << "\n";
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
    return;
} // end of writeData

void assembleSystem(SparseMatrix& M,
                    SparseMatrix& K,
                    double* s,
                    double* T,
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

    if (myID == 0)
    {
        cout << "Assembling system... " << flush;
    }

    double x[nodesPerElement];
    double y[nodesPerElement];
    double z[nodesPerElement];
    double G[numDims][nodesPerElement];

    // for dot product
    double Gp[numDims] = {0.0, 0.0, 0.0};
    double Gq[numDims] = {0.0, 0.0, 0.0};

    double M_e[nodesPerElement][nodesPerElement] = {{2.0, 1.0, 1.0, 1.0},
                                                    {1.0, 2.0, 1.0, 1.0},
                                                    {1.0, 1.0, 2.0, 1.0},
                                                    {1.0, 1.0, 1.0, 2.0}};
    double K_e[nodesPerFace][nodesPerFace] = {
        {2.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 2.0}};
    double s_e[nodesPerElement] = {1.0, 1.0, 1.0, 1.0};
    int Nodes[nodesPerElement]  = {0, 0, 0, 0};

    double* Omega = new double[myN_e];
    double* Gamma = new double[myN_f];
    int m;
    int n;

    for (int p = 0; p < myN_p; p++)
    {
        s[p] = 0.0;
    }

    // Calculate face areas

    for (int f = 0; f < myN_f; f++)
    {

        for (int p = 0; p < nodesPerFace; p++)
        {
            x[p] = Points[Faces[f][p]][0];
            y[p] = Points[Faces[f][p]][1];
            z[p] = Points[Faces[f][p]][2];
        }
        Gamma[f] =
            sqrt(
                pow((y[1] - y[0]) * (z[2] - z[0]) - (z[1] - z[0]) * (y[2] - y[0]), 2.0) +
                pow((z[1] - z[0]) * (x[2] - x[0]) - (x[1] - x[0]) * (z[2] - z[0]), 2.0) +
                pow((x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0]), 2.0)) /
            2.0;
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

        Omega[e] = fabs((x[0] * y[1] * z[2] - x[0] * y[2] * z[1] - x[1] * y[0] * z[2] +
                         x[1] * y[2] * z[0] + x[2] * y[0] * z[1] - x[2] * y[1] * z[0] -
                         x[0] * y[1] * z[3] + x[0] * y[3] * z[1] + x[1] * y[0] * z[3] -
                         x[1] * y[3] * z[0] - x[3] * y[0] * z[1] + x[3] * y[1] * z[0] +
                         x[0] * y[2] * z[3] - x[0] * y[3] * z[2] - x[2] * y[0] * z[3] +
                         x[2] * y[3] * z[0] + x[3] * y[0] * z[2] - x[3] * y[2] * z[0] -
                         x[1] * y[2] * z[3] + x[1] * y[3] * z[2] + x[2] * y[1] * z[3] -
                         x[2] * y[3] * z[1] - x[3] * y[1] * z[2] + x[3] * y[2] * z[1]) /
                        6.0);
    }

    // Assemble M, K, and s

    M.initialize(myN_p, 10); // floor(0.02*myN_p));
    K.initialize(myN_p, 10); // floor(0.02*myN_p));

    for (int e = 0; e < myN_e; e++)
    {
        for (int p = 0; p < nodesPerElement; p++)
        {
            Nodes[p] = Elements[e][p];
            x[p]     = Points[Nodes[p]][0];
            y[p]     = Points[Nodes[p]][1];
            z[p]     = Points[Nodes[p]][2];
        }

        double G[numDims][nodesPerElement] = {
            {(y[3] - y[1]) * (z[2] - z[1]) - (y[2] - y[1]) * (z[3] - z[1]),
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
            m     = Nodes[p];
            Gp[0] = G[0][p];
            Gp[1] = G[1][p];
            Gp[2] = G[2][p];

            // Inner loop over each node
            for (int q = 0; q < nodesPerElement; q++)
            {
                n     = Nodes[q];
                Gq[0] = G[0][q];
                Gq[1] = G[1][q];
                Gq[2] = G[2][q];

                M(m, n) += M_e[p][q] * Omega[e] / 20.0;
                K(m, n) -= alpha_heat * (Gp[0] * Gq[0] + Gp[1] * Gq[1] + Gp[2] * Gq[2]) /
                           (36.0 * Omega[e]);
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
                    m        = Nodes[p];
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
                    m        = Nodes[p];
                    s[m] -= robin_source_constant * Gamma[Boundaries[b].indices_[f]];
                    for (int q = 0; q < nodesPerFace; q++)
                    {
                        Nodes[q] = Faces[Boundaries[b].indices_[f]][q];
                        n        = Nodes[q];
                        K(m, n) += Gamma[Boundaries[b].indices_[f]] *
                                   robin_stiffness_constant * K_e[p][q];
                    }
                }
            }
        }
    }

    K.finalize();
    M.finalize();

    delete[] Gamma;
    delete[] Omega;
    MPI_Barrier(MPI_COMM_WORLD);

    if (myID == 0)
    {
        cout << "Done.\n" << flush;
    }

    return;
} // end of assembleSystem

void solve(SparseMatrix& A,
           double* T,
           double* b,
           Boundary* Boundaries,
           bool* yourPoints,
           int myN_b,
           int myID)
{
    int N_row                 = A.getNrow();
    double* r                 = new double[N_row];
    double* r_old             = new double[N_row];
    double* d                 = new double[N_row];
    double* Ad                = new double[N_row];
    double* AT                = new double[N_row];
    static double alpha       = 0.0;
    static double beta        = 0.0;
    static double r_norm      = 0.0;
    double first_r_norm       = 0.0;
    double tolerance          = 1.0e-8;
    int maxIterations         = 2000;
    int iter                  = 0;
    int k                     = 0;
    int m                     = 0;
    int n                     = 0;
    static double r_oldTr_old = 0.0;
    static double rTr         = 0.0;
    static double dTAd        = 0.0;

    memset(r_old, 0, N_row * sizeof(double));
    memset(r, 0, N_row * sizeof(double));
    memset(d, 0, N_row * sizeof(double));
    memset(Ad, 0, N_row * sizeof(double));

    if (myID == 0)
    {
        cout << "Solving... " << endl;
    }

#pragma omp parallel default(shared)
    {
        // Compute the initial residual
        A.multiply(AT, T);

#pragma omp single
        {
            exchangeData(AT, Boundaries, myN_b);
        }

#pragma omp for private(m)
        for (m = 0; m < N_row; m++)
        {
            r_old[m] = b[m] - AT[m];
            d[m]     = r_old[m];
        }

        r_oldTr_old = computeInnerProduct(r_old, r_old, yourPoints, N_row);

#pragma omp single
        {
            first_r_norm = sqrt(r_oldTr_old);
            r_norm       = 1.0;
        }

        // Conjugate Gradient iterative loop
        while (r_norm > tolerance && iter < maxIterations)
        {
            A.multiply(Ad, d);

#pragma omp single
            {
                exchangeData(Ad, Boundaries, myN_b);
            }

            dTAd = computeInnerProduct(d, Ad, yourPoints, N_row);

#pragma omp single
            {
                alpha = r_oldTr_old / dTAd;
            }

#pragma omp for private(m)
            for (m = 0; m < N_row; m++)
            {
                T[m] += alpha * d[m];
                r[m] = r_old[m] - alpha * Ad[m];
            }

            rTr = computeInnerProduct(r, r, yourPoints, N_row);

#pragma omp single
            {
                beta = rTr / r_oldTr_old;
            }

#pragma omp for private(m)
            for (m = 0; m < N_row; m++)
            {
                d[m]     = r[m] + beta * d[m];
                r_old[m] = r[m];
            }

#pragma omp single
            {
                r_oldTr_old = rTr;

                r_norm = std::sqrt(rTr) / first_r_norm;
                iter++;
            }

        } // end of while loop

#pragma omp single
        {
            if (myID == 0)
            {
                cout << ", iter = " << iter << ", r_norm = " << r_norm << endl;
            }

            delete[] r_old;
            delete[] r;
            delete[] d;
            delete[] Ad;
            delete[] AT;
        }

    } // end of pragma omp parallel

    return;

} // end of solve

double computeInnerProduct(double* v1, double* v2, bool* yourPoints, int N_row)
{
    static double myInnerProduct;
    static double innerProduct;

    myInnerProduct = 0.0;
    innerProduct   = 0.0;

    int m;

#pragma omp for schedule(static) reduction(+ : myInnerProduct)
    for (int m = 0; m < N_row; m++)
    {
        if (!yourPoints[m])
        {
            myInnerProduct += v1[m] * v2[m];
        }
    }

#pragma omp single
    {
        MPI_Allreduce(
            &myInnerProduct, &innerProduct, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    return innerProduct;
}
