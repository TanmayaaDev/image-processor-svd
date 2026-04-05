#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

int main() {
    int width, height, channels;
    
    // 1. Load as Grayscale
    unsigned char *img = stbi_load("ambition_roadmap.png", &width, &height, &channels, 1);
    if (img == NULL) {
        printf("Error: Could not load image.\n");
        return 1;
    }

    // 2. Allocate Matrix
    gsl_matrix *A = gsl_matrix_alloc(height, width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            gsl_matrix_set(A, i, j, (double)img[i * width + j]);
        }
    }

    // 3. Setup Workspace (Jacobi only needs V and S)
    gsl_matrix *V = gsl_matrix_alloc(width, width);
    gsl_vector *S = gsl_vector_alloc(width);

    // 4. Perform Jacobi SVD (Works for wide AND tall images)
    printf("Processing SVD (this may take a moment for wide images)...\n");
    gsl_linalg_SV_decomp_jacobi(A, V, S);

    // 5. Reconstruct with Rank k
    int k = 100; // Start with 100 for a balance of quality/size
    unsigned char *out_img = malloc(width * height);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double sum = 0;
            // Reconstruction: U * S * V^T
            for (int m = 0; m < k; m++) {
                double u_im = gsl_matrix_get(A, i, m);
                double s_m  = gsl_vector_get(S, m);
                double v_jm = gsl_matrix_get(V, j, m); 
                sum += u_im * s_m * v_jm;
            }
            // Clamp 0-255
            if (sum > 255) sum = 255;
            if (sum < 0) sum = 0;
            out_img[i * width + j] = (unsigned char)sum;
        }
    }

    // 6. Save
    stbi_write_png("grayscale_compressed.png", width, height, 1, out_img, width);

    // 7. Cleanup
    gsl_matrix_free(A);
    gsl_matrix_free(V);
    gsl_vector_free(S);
    free(out_img);
    stbi_image_free(img);

    printf("Success! Saved as grayscale_compressed.png\n");
    return 0;
}