//
// Created by Nicolae Marius Ghergu on 18.04.2023.
//

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 200

// functie pentru citirea datelor din fisier
void read_data(float *x, float *y, float *a) {
    FILE *x_file = fopen("x.dat", "r");
    FILE *y_file = fopen("y.dat", "r");
    FILE *mat_file = fopen("mat.dat", "r");

    for (int i = 0; i < N; i++) {
        fscanf(x_file, "%f", &x[i]);
        fscanf(y_file, "%f", &y[i]);
        for (int j = 0; j < N; j++) {
            fscanf(mat_file, "%f", &a[i * N + j]);
        }
    }

    fclose(x_file);
    fclose(y_file);
    fclose(mat_file);
}

// functie pentru scrierea rezultatului in fisier
void write_result(float result) {
    FILE *output_file = fopen("output.txt", "w");
    fprintf(output_file, "AVG = %f\n", result);
    fclose(output_file);
}

int main(int argc, char *argv[]) {
    int my_rank, nproc, my_new_rank;
    float *x, *y, *a;
    float *x_loc, *y_loc, *a_loc;
    float numerator, numerator_loc = 0.0, denominator, denominator_loc = 0.0;

    // initializare MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // nproc este par
    if (nproc % 2 != 0) {
        if (my_rank == 0) {
            printf("Numarul de procese trebuie sa fie par.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Crearea grupurilor si a comunicatoarelor

    // Alocare memorie pentru vectori si matrice
    int nproc_per_group = nproc / 2;
    int elements_per_process = N / nproc_per_group;
    x_loc = (float *)malloc(elements_per_process * sizeof(float));
    y_loc = (float *)malloc(elements_per_process * sizeof(float));
    a_loc = (float *)malloc(elements_per_process * N * sizeof(float));

    if (my_new_rank == 0) {
        x = (float *)malloc(N * sizeof(float));
        y = (float *)malloc(N * sizeof(float));
        a = (float *)malloc(N * N * sizeof(float));
        read_data(x, y, a);
    }

    // distribuie datele in procesele din fiecare grup
    if (my_rank < nproc_per_group) {
        // grupul low_group
        MPI_Scatter(x, elements_per_process, MPI_FLOAT, x_loc, elements_per_process, MPI_FLOAT, 0, low_comm);
        MPI_Scatter(y, elements_per_process, MPI_FLOAT, y_loc, elements_per_process, MPI_FLOAT, 0, low_comm);
        MPI_Scatter(a, elements_per_process * N, MPI_FLOAT, a_loc, elements_per_process * N, MPI_FLOAT, 0, low_comm);

        // calculul local al numaratorului pentru fiecare proces din grupul low_group
        for (int i = 0; i < elements_per_process; i++) {
            for (int j = 0; j < N; j++) {
                numerator_loc += x_loc[i] * a_loc[i * N + j] * y_loc[j];
            }
        }

        // combinarea rezultatelor locale ale numaratorului in procesul cu my_new_rank = 0
        MPI_Reduce(&numerator_loc, &numerator, 1, MPI_FLOAT, MPI_SUM, 0, low_comm);
    } else {
        // grupul high_group
        MPI_Scatter(x, elements_per_process, MPI_FLOAT, x_loc, elements_per_process, MPI_FLOAT, 0, high_comm);
        MPI_Scatter(y, elements_per_process, MPI_FLOAT, y_loc, elements_per_process, MPI_FLOAT, 0, high_comm);
        MPI_Scatter(a, elements_per_process * N, MPI_FLOAT, a_loc, elements_per_process * N, MPI_FLOAT, 0, high_comm);

        // calculul local al numitorului pentru fiecare proces din grupul high_group
        for (int i = 0; i < elements_per_process; i++) {
            for (int j = 0; j < N; j++) {
                denominator_loc += x_loc[i] * a_loc[i * N + j] * y_loc[j] * x_loc[i] * a_loc[i * N + j] * y_loc[j];
            }
        }

        // combinarea rezultatelor locale ale numitorului in procesul cu my_new_rank = 0
        MPI_Reduce(&denominator_loc, &denominator, 1, MPI_FLOAT, MPI_SUM, 0, high_comm);

        // procesul cu my_rank >= nproc/2 si my_new_rank = 0 trimite rezultatul (numitorul) catre procesul cu my_rank = 0
        if (my_new_rank == 0) {
            MPI_Send(&denominator, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    // procesul cu my_rank = 0 primeste valoarea numitorului, calculeaza rezultatul si il scrie intr-un fișier
    if (my_rank == 0) {
        MPI_Recv(&denominator, 1, MPI_FLOAT, nproc_per_group, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        float result = numerator / denominator;
        write_result(result);
    }

    // eliberare memorie alocata
    free(x_loc);
    free(y_loc);
    free(a_loc);

    if (my_new_rank == 0) {
        free(x);
        free(y);
        free(a);
    }

    // eliberare grupuri si comunicatoare
    MPI_Group_free(&world_group);
    MPI_Group_free(&low_group);
    MPI_Group_free(&high_group);
    if (low_comm != MPI_COMM_NULL) MPI_Comm_free(&low_comm);
    if (high_comm != MPI_COMM_NULL) MPI_Comm_free(&high_comm);

    // finalizare MPI
    MPI_Finalize();
    return 0;
}