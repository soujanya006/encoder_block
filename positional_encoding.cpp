#include "positional_encoding.h"
#include <hls_math.h>

//positional logic


// down here

//positional logic

void positional_encoding(float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE]) {
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=pos_enc bundle=control

    const float32_t div_term = 1e4;

    float32_t angle_rates[EMBEDDING_SIZE];
    for (int i = 0; i < EMBEDDING_SIZE; i += 2) {
        angle_rates[i] = 1 / hls::powf(div_term, 2 * i / (float32_t)EMBEDDING_SIZE);
        if (i + 1 < EMBEDDING_SIZE) {
            angle_rates[i + 1] = 1 / hls::powf(div_term, 2 * (i + 1) / (float32_t)EMBEDDING_SIZE);
        }
    }

#pragma HLS ARRAY_PARTITION variable=pos_enc dim=1 complete
#pragma HLS ARRAY_PARTITION variable=pos_enc dim=2 complete
    for (int i = 0; i < SEQ_LENGTH; ++i) {
        for (int j = 0; j < EMBEDDING_SIZE; j += 2) {
#pragma HLS UNROLL
            float32_t angle = i * angle_rates[j];
            pos_enc[i][j] = hls::sinf(angle);
            if (j + 1 < EMBEDDING_SIZE) {
                pos_enc[i][j + 1] = hls::cosf(angle);
            }
        }
    }
}


void add_positional_encoding(float32_t input_seq[SEQ_LENGTH][EMBEDDING_SIZE],
float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE], float32_t output_seq[SEQ_LENGTH][EMBEDDING_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=input_seq dim=1 complete
#pragma HLS ARRAY_PARTITION variable=input_seq dim=2 complete
#pragma HLS ARRAY_PARTITION variable=pos_enc dim=1 complete
#pragma HLS ARRAY_PARTITION variable=pos_enc dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output_seq dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output_seq dim=2 complete

    for (int i = 0; i < SEQ_LENGTH; ++i) {
        for (int j = 0; j < EMBEDDING_SIZE; ++j) {
#pragma HLS PIPELINE
            output_seq[i][j] = input_seq[i][j] + pos_enc[i][j];
        }
    }
}





//linear layer logic

// down here


//linear layer logic

#define SEQ_LEN 4
#define IN_DIM 5
#define OUT_DIM 15



template <typename T>
void linear_layer(T input[SEQ_LEN][IN_DIM], T key[SEQ_LEN][OUT_DIM/3],
                  T query[SEQ_LEN][OUT_DIM/3], T value[SEQ_LEN][OUT_DIM/3],
                  T weights[IN_DIM][OUT_DIM], T bias[OUT_DIM]) {
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=5 dim=2
    #pragma HLS ARRAY_PARTITION variable=key complete dim=1
    #pragma HLS ARRAY_PARTITION variable=query complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value complete dim=1
    for (int i = 0; i < SEQ_LEN; i++) {
	#pragma HLS PIPELINE II=3

        T sum[OUT_DIM];
        #pragma HLS ARRAY_PARTITION variable=sum complete dim=1
        for (int j = 0; j < OUT_DIM; j++) {
            #pragma HLS UNROLL
            sum[j] = bias[j];
            for (int k = 0; k < IN_DIM; k++) {
                #pragma HLS UNROLL
                sum[j] += input[i][k] * weights[k][j];
            }
            if (j < OUT_DIM/3) {
                key[i][j] = sum[j];
            } else if (j < 2 * OUT_DIM/3) {
                query[i][j - OUT_DIM/3] = sum[j];
            } else {
                value[i][j - 2 * OUT_DIM/3] = sum[j];
            }
        }
    }
}

void transformer_linear_layer(float32_t input[SEQ_LEN][IN_DIM],
		float32_t key[SEQ_LEN][OUT_DIM/3],
		float32_t query[SEQ_LEN][OUT_DIM/3],
		float32_t value[SEQ_LEN][OUT_DIM/3]) {
	float32_t output[SEQ_LEN][OUT_DIM];

    // Predefined weights and bias based on SEQ_LEN (4) and IN_DIM (5)
	float32_t weights[IN_DIM][OUT_DIM] = {
            {.01, .02, .03, .04, .05, .06, .07, .08, .09, .010, .011, .012, .013, .014, .015},
            {.02, .03, .04, .05, .06, .07, .08, .09, .010, .011, .012, .013, .014, .015, .01},
            {.03, .04, .05, .06, .07, .08, .09, .010, .011, .012, .013, .014, .015, .01, .02},
            {.04, .05, .06, .07, .08, .09, .010, .011, .012, .013, .014, .015, .01, .02, .03},
            {.05, .06, .07, .08, .09, .010, .011, .012, .013, .014, .015, .01, .02, .03, .04}
    };

	float32_t bias[OUT_DIM] = {.01, .02, .03, .04, .05, .06, .07, .08, .09, .010, .011, .012, .013, .014, .015};

    // Call the linear layer function

    linear_layer(input, key,query,value, weights, bias);
    // Call the function to split the output
    //split_output(output, key, query, value);
  }



//self attention block



//
void self_attention(float32_t key[4][5], float32_t query[4][5], float32_t value[4][5], float32_t output_matrix[4][5]) {
   /*

    #pragma HLS INTERFACE s_axilite port=key_matrix bundle=control
    #pragma HLS INTERFACE s_axilite port=query_matrix bundle=control
    #pragma HLS INTERFACE s_axilite port=value_matrix bundle=control
    #pragma HLS INTERFACE s_axilite port=output_matrix bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
*/
    const int seq_length = 4;
    const int dim = 5;
    const float32_t scaling_factor = hls::sqrtf(dim);

    float32_t attention_scores[4][4];
    float32_t softmax_scores[4];

    // Calculate attention scores
    attention_scores_calculation:
    for (int i = 0; i < seq_length; i++) {
        #pragma HLS UNROLL
        for (int j = 0; j < seq_length; j++) {
            #pragma HLS UNROLL
        	float32_t dot_product = 0;
            for (int k = 0; k < dim; k++) {
                #pragma HLS UNROLL
                dot_product += query[i][k] * key[j][k];
            }
            attention_scores[i][j] = dot_product / scaling_factor;
        }
    }

    // Apply softmax and calculate weighted values
    weighted_values_calculation:
    for (int i = 0; i < seq_length; i++) {
        #pragma HLS UNROLL
        softmax(attention_scores[i], softmax_scores);

        float32_t weighted_values[4][5];

        for (int j = 0; j < seq_length; j++) {
            #pragma HLS UNROLL
            for (int k = 0; k < dim; k++) {
                #pragma HLS UNROLL
                weighted_values[j][k] = softmax_scores[j] * value[j][k];
            }
        }

        // Sum the weighted values
        for (int j = 0; j < dim; j++) {
            #pragma HLS UNROLL
        	float32_t sum = 0;
            for (int k = 0; k < seq_length; k++) {
                #pragma HLS UNROLL
                sum += weighted_values[k][j];
            }
            output_matrix[i][j] = sum;
        }
    }
}

void softmax(float32_t input_matrix[4], float32_t output_matrix[4]) {
	float32_t max_val = input_matrix[0];
    max_val_calculation:
    for (int i = 1; i < 4; i++) {
        #pragma HLS UNROLL
        if (input_matrix[i] > max_val) max_val = input_matrix[i];
    }

    float32_t sum_exp = 0;
    sum_exp_calculation:
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        output_matrix[i] = hls::expf(input_matrix[i] - max_val);
        sum_exp += output_matrix[i];
    }

    normalization:
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        output_matrix[i] /= sum_exp;
    }
}



