#include <iostream>
#include "positional_encoding.h"
#include <iomanip>

//positional block

void print_positional_encoding(float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE]) {
    for (int i = 0; i < SEQ_LENGTH; ++i) {
        for (int j = 0; j < EMBEDDING_SIZE; ++j) {
            std::cout << static_cast<float>(pos_enc[i][j]) << " ";
                                                           }
        std::cout << std::endl;
    }
}

// linear layer block

#define ERROR_THRESHOLD 0.001

void print_matrix(float32_t matrix[][OUT_DIM / 3], const char *name) {
    std::cout << name << " matrix:" << std::endl;
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < OUT_DIM / 3; j++) {
            std::cout << static_cast<float>(matrix[i][j]) << " ";
        }
        std::cout << std::endl;
    }
}


// self attention block

//
void self_attention_print(float32_t output_matrix[][5])
{
	std::cout << "Output matrix:" << std::endl;
	    for (int i = 0; i < 4; i++) {
	        for (int j = 0; j < 5; j++) {
	            std::cout << static_cast<float>(output_matrix[i][j]) << " ";
	        }
	        std::cout << std::endl;
	    }

}



//MAIN FUNCTION

int main() {
    float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE];
    float32_t output[SEQ_LENGTH][EMBEDDING_SIZE];

    //input of the transformer as positional encoding
    float32_t custom_values[SEQ_LENGTH][EMBEDDING_SIZE] = {
        {0.1, 0.2, 0.3, 0.4, 0.5},
        {0.7, 0.8, 0.9, 1.0, 1.1},
        {1.3, 1.4, 1.5, 1.6, 1.7},
        {1.9, 2.0, 2.1, 2.2, 2.3},

    };

    //block 1 positional encoding call

    //
    positional_encoding(pos_enc);
    std::cout << "Positional Encoding: " << std::endl;
    print_positional_encoding(pos_enc);

    std::cout << "Final added value: " << std::endl;
    //Block one
    add_positional_encoding(custom_values,pos_enc,output);
    print_positional_encoding(output);



    //block 2  linear layer for key query value ¨



    //



    // Call the top-level function
    float32_t key[SEQ_LEN][OUT_DIM / 3];
    float32_t query[SEQ_LEN][OUT_DIM / 3];
    float32_t value[SEQ_LEN][OUT_DIM / 3];

    transformer_linear_layer(output, key, query, value);
    // Print the results
    print_matrix(key, "Key");
    print_matrix(query, "Query");
    print_matrix(value, "Value");


    ///self attention block

    //

    float32_t output_matrix[4][5];

    self_attention(key, query, value, output_matrix);

    self_attention_print(output_matrix);


    return 0;

}
