#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int H, W, F;

#define C 3
#define K 3

float *input;
float *kernel;
float *output;

#define INPUT(c,i,j) input[(c)*H*W + (i)*W + (j)]
#define KERNEL(f,c,i,j) kernel[(f)*C*K*K + (c)*K*K + (i)*K + (j)]
#define OUTPUT(f,i,j) output[(f)*(H-K+1)*(W-K+1) + (i)*(W-K+1) + (j)]

// ---------------- LOAD IMAGE ----------------
void load_image() {
    FILE *fp = fopen("image.bin", "rb");
    if (!fp) {
        printf("Error loading image\n");
        exit(1);
    }
    fread(input, sizeof(float), C*H*W, fp);
    fclose(fp);
}

// ---------------- INIT KERNEL ----------------
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

// ---------------- NAIVE ----------------
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

// ---------------- OPENMP ----------------
void openmp_conv() {
#pragma omp parallel for collapse(3)
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

// ---------------- IMG2COL SERIAL ----------------
void img2col_serial() {

    int outH = H-K+1;
    int outW = W-K+1;
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

// ---------------- IMG2COL OPENMP ----------------
void img2col_openmp() {

    int outH = H-K+1;
    int outW = W-K+1;
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

#pragma omp parallel for collapse(2)
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

// ---------------- MAIN ----------------
int main(int argc, char *argv[]) {

    if (argc != 4) {
        printf("Usage: ./run <size> <filters> <threads>\n");
        return 1;
    }

    H = atoi(argv[1]);
    W = H;
    F = atoi(argv[2]);
    int threads = atoi(argv[3]);

    omp_set_num_threads(threads);

    printf("\n===== Size=%d Filters=%d Threads=%d =====\n", H, F, threads);

    input  = malloc(C * H * W * sizeof(float));
    kernel = malloc(F * C * K * K * sizeof(float));
    output = malloc(F * (H-K+1) * (W-K+1) * sizeof(float));

    load_image();
    init_kernel();

    double start, end;

    double naive_time, openmp_time, img2col_serial_time, img2col_openmp_time;

    // Naive
    start = omp_get_wtime();
    naive_conv();
    end = omp_get_wtime();
    naive_time = end - start;
    printf("Naive Time: %f\n", naive_time);

    // OpenMP
    start = omp_get_wtime();
    openmp_conv();
    end = omp_get_wtime();
    openmp_time = end - start;
    printf("OpenMP Time: %f\n", openmp_time);

    // img2col Serial
    start = omp_get_wtime();
    img2col_serial();
    end = omp_get_wtime();
    img2col_serial_time = end - start;
    printf("img2col Serial Time: %f\n", img2col_serial_time);

    // img2col OpenMP
    start = omp_get_wtime();
    img2col_openmp();
    end = omp_get_wtime();
    img2col_openmp_time = end - start;
    printf("img2col OpenMP Time: %f\n", img2col_openmp_time);

    // -------- SAVE TO FILE --------
    // Add at top or inside main
    int save = 0;   // 0 = don't save, 1 = save


    if (save) {
        FILE *fp = fopen("results2.txt", "a");
        if (fp != NULL) {
            fprintf(fp, "%d %d %d %f %f %f %f\n",
                    H, F, threads,
                    naive_time,
                    openmp_time,
                    img2col_serial_time,
                    img2col_openmp_time);
            fclose(fp);
        }
    }

    free(input);
    free(kernel);
    free(output);

    return 0;
}