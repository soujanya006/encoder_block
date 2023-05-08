//  positional h code


//
#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H
#include <hls_stream.h>
#include <ap_fixed.h>

#include <cmath>
#include <vector>

#define SEQ_LENGTH 4
#define EMBEDDING_SIZE 5

typedef ap_fixed<32, 12> float32_t;




void positional_encoding(float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE]);


void add_positional_encoding(float32_t custom_values[SEQ_LENGTH][EMBEDDING_SIZE],
		float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE],
		float32_t output[SEQ_LENGTH][EMBEDDING_SIZE]);


//  transformer linear layer h codes

//
#define SEQ_LEN 4
#define IN_DIM 5
#define OUT_DIM 15

// Top-level function for the transformer linear layer
void transformer_linear_layer(float32_t input[SEQ_LEN][IN_DIM],
		float32_t key[SEQ_LEN][OUT_DIM/3],
		float32_t query[SEQ_LEN][OUT_DIM/3],
		float32_t value[SEQ_LEN][OUT_DIM/3]);



//self attention

void self_attention(float32_t key[4][5], float32_t query[4][5], float32_t value[4][5], float32_t output_matrix[4][5]);
void softmax(float32_t input_matrix[4], float32_t output_matrix[4]);


#endif


