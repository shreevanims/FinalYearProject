#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

int H, W, F;

#define C 3
#define K 3

float *input;
float *kernel;
float *output;

#define INPUT(c,i,j) input[(c)*H*W + (i)*W + (j)]
#define KERNEL(f,c,i,j) kernel[(f)*C*K*K + (c)*K*K + (i)*K + (j)]
#define OUTPUT(f,i,j) output[(f)*(H-K+1)*(W-K+1) + (i)*(W-K+1) + (j)]

/* ---------------- LOAD IMAGE ---------------- */
void load_image() {
    FILE *fp = fopen("image.bin", "rb");
    if (!fp) {
        printf("Error: image.bin not found!\n");
        exit(1);
    }

    size_t read = fread(input, sizeof(float), C*H*W, fp);
    if (read != C*H*W) {
        printf("Error: image.bin size mismatch!\n");
        exit(1);
    }

    fclose(fp);
}

/* ---------------- INIT KERNEL ---------------- */
void init_kernel() {
    float sharpen[K][K] = {
        {0,-1,0},
        {-1,5,-1},
        {0,-1,0}
    };

    for (int f=0; f<F; f++)
        for (int c=0; c<C; c++)
            for (int i=0; i<K; i++)
                for (int j=0; j<K; j++)
                    KERNEL(f,c,i,j) = sharpen[i][j];
}

/* ---------------- NAIVE ---------------- */
void naive_conv() {
    for (int f=0; f<F; f++)
        for (int i=0; i<H-K+1; i++)
            for (int j=0; j<W-K+1; j++) {

                float sum = 0;

                for (int c=0; c<C; c++)
                    for (int kh=0; kh<K; kh++)
                        for (int kw=0; kw<K; kw++)
                            sum += INPUT(c,i+kh,j+kw) * KERNEL(f,c,kh,kw);

                OUTPUT(f,i,j) = sum;
            }
}

/* ---------------- OPENMP CONV ---------------- */
void openmp_conv(char *schedule_type, int chunk) {

    if (strcmp(schedule_type, "static") == 0) {

#pragma omp parallel for collapse(3) schedule(static, chunk)
        for (int f=0; f<F; f++)
            for (int i=0; i<H-K+1; i++)
                for (int j=0; j<W-K+1; j++) {

                    float sum = 0;
                    for (int c=0; c<C; c++)
                        for (int kh=0; kh<K; kh++)
                            for (int kw=0; kw<K; kw++)
                                sum += INPUT(c,i+kh,j+kw) * KERNEL(f,c,kh,kw);

                    OUTPUT(f,i,j) = sum;
                }

    } else if (strcmp(schedule_type, "dynamic") == 0) {

#pragma omp parallel for collapse(3) schedule(dynamic, chunk)
        for (int f=0; f<F; f++)
            for (int i=0; i<H-K+1; i++)
                for (int j=0; j<W-K+1; j++) {

                    float sum = 0;
                    for (int c=0; c<C; c++)
                        for (int kh=0; kh<K; kh++)
                            for (int kw=0; kw<K; kw++)
                                sum += INPUT(c,i+kh,j+kw) * KERNEL(f,c,kh,kw);

                    OUTPUT(f,i,j) = sum;
                }

    } else if (strcmp(schedule_type, "guided") == 0) {

#pragma omp parallel for collapse(3) schedule(guided, chunk)
        for (int f=0; f<F; f++)
            for (int i=0; i<H-K+1; i++)
                for (int j=0; j<W-K+1; j++) {

                    float sum = 0;
                    for (int c=0; c<C; c++)
                        for (int kh=0; kh<K; kh++)
                            for (int kw=0; kw<K; kw++)
                                sum += INPUT(c,i+kh,j+kw) * KERNEL(f,c,kh,kw);

                    OUTPUT(f,i,j) = sum;
                }

    } else {
        printf("Invalid schedule type\n");
    }
}

/* ---------------- IMG2COL SERIAL ---------------- */
void img2col_serial() {

    int outH = H-K+1, outW = W-K+1;
    int rows = outH * outW;
    int cols = C*K*K;

    float *im2col = malloc(rows * cols * sizeof(float));
    float *kernel_mat = malloc(cols * F * sizeof(float));

    int idx = 0;
    for (int i=0; i<outH; i++)
        for (int j=0; j<outW; j++) {
            int col = 0;
            for (int c=0; c<C; c++)
                for (int kh=0; kh<K; kh++)
                    for (int kw=0; kw<K; kw++)
                        im2col[idx*cols + col++] = INPUT(c,i+kh,j+kw);
            idx++;
        }

    for (int f=0; f<F; f++)
        for (int c=0; c<C; c++)
            for (int kh=0; kh<K; kh++)
                for (int kw=0; kw<K; kw++) {
                    int row = c*K*K + kh*K + kw;
                    kernel_mat[row*F + f] = KERNEL(f,c,kh,kw);
                }

    for (int i=0; i<rows; i++)
        for (int j=0; j<F; j++) {
            float sum = 0;
            for (int k=0; k<cols; k++)
                sum += im2col[i*cols + k] * kernel_mat[k*F + j];

            int r = i / outW;
            int c2 = i % outW;
            OUTPUT(j,r,c2) = sum;
        }

    free(im2col);
    free(kernel_mat);
}

/* ---------------- IMG2COL OPENMP ---------------- */
void img2col_openmp(char *schedule_type, int chunk) {

    int outH = H-K+1, outW = W-K+1;
    int rows = outH * outW;
    int cols = C*K*K;

    float *im2col = malloc(rows * cols * sizeof(float));
    float *kernel_mat = malloc(cols * F * sizeof(float));

    int idx = 0;
    for (int i=0; i<outH; i++)
        for (int j=0; j<outW; j++) {
            int col = 0;
            for (int c=0; c<C; c++)
                for (int kh=0; kh<K; kh++)
                    for (int kw=0; kw<K; kw++)
                        im2col[idx*cols + col++] = INPUT(c,i+kh,j+kw);
            idx++;
        }

    for (int f=0; f<F; f++)
        for (int c=0; c<C; c++)
            for (int kh=0; kh<K; kh++)
                for (int kw=0; kw<K; kw++) {
                    int row = c*K*K + kh*K + kw;
                    kernel_mat[row*F + f] = KERNEL(f,c,kh,kw);
                }

    if (strcmp(schedule_type, "static") == 0) {

#pragma omp parallel for collapse(2) schedule(static, chunk)
        for (int i=0; i<rows; i++)
            for (int j=0; j<F; j++) {

                float sum = 0;
                for (int k=0; k<cols; k++)
                    sum += im2col[i*cols + k] * kernel_mat[k*F + j];

                int r = i / outW;
                int c2 = i % outW;
                OUTPUT(j,r,c2) = sum;
            }

    } else if (strcmp(schedule_type, "dynamic") == 0) {

#pragma omp parallel for collapse(2) schedule(dynamic, chunk)
        for (int i=0; i<rows; i++)
            for (int j=0; j<F; j++) {

                float sum = 0;
                for (int k=0; k<cols; k++)
                    sum += im2col[i*cols + k] * kernel_mat[k*F + j];

                int r = i / outW;
                int c2 = i % outW;
                OUTPUT(j,r,c2) = sum;
            }

    } else if (strcmp(schedule_type, "guided") == 0) {

#pragma omp parallel for collapse(2) schedule(guided, chunk)
        for (int i=0; i<rows; i++)
            for (int j=0; j<F; j++) {

                float sum = 0;
                for (int k=0; k<cols; k++)
                    sum += im2col[i*cols + k] * kernel_mat[k*F + j];

                int r = i / outW;
                int c2 = i % outW;
                OUTPUT(j,r,c2) = sum;
            }

    } else {
        printf("Invalid schedule type\n");
    }

    free(im2col);
    free(kernel_mat);
}


// void print_dimensions() {

//     int outH = H - K + 1;
//     int outW = W - K + 1;

//     int rows = outH * outW;
//     int cols = C * K * K;

//     printf("\n=========== MATRIX DIMENSIONS ===========\n");

//     printf("Input Tensor: %d x %d x %d (C x H x W)\n", C, H, W);

//     printf("Kernel Tensor: %d x %d x %d x %d (F x C x K x K)\n", F, C, K, K);

//     printf("Output Tensor: %d x %d x %d (F x H_out x W_out)\n", F, outH, outW);

//     printf("\n--- IM2COL TRANSFORMATION ---\n");
//     printf("im2col Matrix: (%d x %d)\n", rows, cols);

//     printf("Kernel Matrix: (%d x %d)\n", cols, F);

//     printf("\n--- MATRIX MULTIPLICATION ---\n");
//     printf("(%d x %d) x (%d x %d) = (%d x %d)\n",
//            rows, cols, cols, F, rows, F);

//     printf("\n=========================================\n\n");
// }

/* ---------------- MAIN ---------------- */
int main(int argc, char *argv[]) {

    if (argc != 6) {
        printf("Usage: ./run <size> <filters> <threads> <schedule> <chunk>\n");
        return 1;
    }

    H = atoi(argv[1]);
    W = H;
    F = atoi(argv[2]);
    int threads = atoi(argv[3]);
    char *schedule_type = argv[4];
    int chunk = atoi(argv[5]);

    omp_set_num_threads(threads);

    input  = malloc(C * H * W * sizeof(float));
    kernel = malloc(F * C * K * K * sizeof(float));
    output = malloc(F * (H-K+1) * (W-K+1) * sizeof(float));

    load_image();
    init_kernel();

    // print_dimensions();

    double t1, t2, t3, t4, start, end;

    start = omp_get_wtime(); naive_conv(); end = omp_get_wtime(); t1 = end-start;
    start = omp_get_wtime(); openmp_conv(schedule_type, chunk); end = omp_get_wtime(); t2 = end-start;
    start = omp_get_wtime(); img2col_serial(); end = omp_get_wtime(); t3 = end-start;
    start = omp_get_wtime(); img2col_openmp(schedule_type, chunk); end = omp_get_wtime(); t4 = end-start;

    printf("\nNaive: %.6f | OpenMP: %.6f | im2col S: %.6f | im2col O: %.6f\n", t1,t2,t3,t4);

    FILE *fp = fopen("results2.txt", "a");
    if (fp) {
        fprintf(fp,"%d %d %d %s %d %.6f %.6f %.6f %.6f\n",
                H,F,threads,schedule_type,chunk,t1,t2,t3,t4);
        fclose(fp);
    }

    free(input); free(kernel); free(output);
    return 0;
}