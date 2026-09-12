import sys
sys.path.append('/usr/local')

import openfhe

def create_crypto_context(depth, batch_size):
    parameters = openfhe.CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(depth)
    parameters.SetScalingModSize(50)
    parameters.SetBatchSize(batch_size)

    cc = openfhe.GenCryptoContext(parameters)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.ADVANCEDSHE)

    return cc

def gen_keys(context):
    keys = context.KeyGen()
    context.EvalMultKeyGen(keys.secretKey)
    context.EvalAtIndexKeyGen(keys.secretKey, [1, 2, 4, 8, -1, -2, -3])

    return keys

def get_data_weight():
    data = [0.5, 0.8, -0.2, 1.1, 3.0, 4.6, 0.6, 1]

    # Matrix of weights for 3 neurons (each with 8 weights)
    weights_matrix = [
        [1.0, -1.0, 0.5, 2.0, 1.0, 1.5, 2.0, 1.0],  # Neuron 0
        [0.5,  0.5, 1.0, 1.0, 0.0, 2.0, 1.5, 0.5],  # Neuron 1
        [-1.0, 2.0, 0.0, 1.0, 1.0, 1.0, -0.5, 1.0]  # Neuron 2
    ]
    
    return data, weights_matrix

def enc(message, context, keys):
    ptxt_message = context.MakeCKKSPackedPlaintext(message)
    ctxt_message = context.Encrypt(keys.publicKey, ptxt_message)

    return ctxt_message

def dec_and_print(ctxt, keys, openFheContext):
    res = openFheContext.Decrypt(ctxt, keys.secretKey)
    print(res)

def eval_sum_tree(context, ctxt, vector_size):
    shift = 1

    while shift < vector_size:
        # create a rotate copy of the ciphertext
        ctxt_rot = context.EvalAtIndex(ctxt, shift)

        ctxt_in_func = context.EvalAdd(ctxt, ctxt_rot)         

        # Doubles the shift
        shift *=2

    return ctxt_in_func

def eval_layer(context, ctxt_packed_data, ctxt_packed_weights, block_size):
    # Dot product for entire layer sunyktaneously
    ctxt_mult = context.EvalMult(ctxt_packed_data, ctxt_packed_weights)

    # Sum Tree
    ctxt_sum_tree = eval_sum_tree(context, ctxt_mult, block_size)

    # Approximation of Sigmpid Function (-0.004*x^3 + 0.197*x + 0.5)
    coefficients = [0.5, 0.197, 0.0, -0.004]

    # Evaluate polynomial homomorphically
    ctxt_activation = context.EvalPoly(ctxt_sum_tree, coefficients)
    
    return ctxt_activation

def pack_data_for_layer(data, weights_matrix, num_neurons):
    input_len = len(data)
    
    # Repeat the input data vector for each neuron
    # [0.5, 0.8] -> [0.5, 0.8, 0.5, 0.8] for 2 neurons
    packed_data = []
    for _ in range(num_neurons):
        packed_data.extend(data)

    # Transform the weights matrix into a 1D array
    packed_weights = []
    for neuron_weights in weights_matrix:
        if len(neuron_weights) != input_len:
            raise ValueError("Weight vector size must match input data size")
        packed_weights.extend(neuron_weights)
        
    return packed_data, packed_weights

def main():
    # Instantiate the cryptographic context
    cc = create_crypto_context(depth=4, batch_size=32)

    # Generate keys (public, private, and rotation/multiplication keys)
    keys = gen_keys(cc)

    # Load inputs and weights
    data, weights_matrix = get_data_weight()

    num_neurons = len(weights_matrix)

    # Pack data and weights_matrix
    print("Packing data and weights for SIMD execution...")
    p_data, p_weights = pack_data_for_layer(data, weights_matrix, num_neurons)

    # Encrypt data & Weights
    ctxt_packed_data = enc(p_data, cc, keys)
    ctxt_packed_weights = enc(p_weights, cc, keys)

    # Evaluate entire layer (mult + tree + activation)
    print("Evaluating the entire dense layer (SIMD)...")
    ctxt_layer_out = eval_layer(cc, ctxt_packed_data, ctxt_packed_weights, len(data))

    print("Decrypting final layer output...")
    ptxt_res = cc.Decrypt(ctxt_layer_out, keys.secretKey)

    # Extract values of plaintext
    ptxt_res.SetLength(num_neurons * len(data)) 
    res_array = ptxt_res.GetRealPackedValue()

    print("\n=== Final Clean Results ===")
    for i in range(num_neurons):
        valid_index = i * len(data)
        val = res_array[valid_index]
        print(f"Neuron {i} output (Index {valid_index}): {val:.4f}")

    # Homomorphic multiplication (data * weight)
    # print("Performing homomorphic element-wise multiplication...")
    # ctxt_mult = cc.EvalMult(ctxt_data, ctxt_weights)

    # Check multiplication
    # print("Multiplication result (input for the Sum Tree):")
    # dec_and_print(ctxt_mult, keys, cc)

    # Apply Sum Tree Function
    # print("\nExecuting Sum Tree...")
    # ctxt_sum_tree_res = eval_sum_tree(cc, ctxt_mult, len(data))

    # print("Final Sum Tree result (Total sum is at index 0):")
    # dec_and_print(ctxt_sum_tree_res, keys, cc)

    # Single Neuron Test
    # print("Executing Single Neuron (Dot Product + Sigmoid Approximation)...")
    # ctxt_neuron_result = eval_single_neuron(cc, ctxt_data, ctxt_weights, len(data))
    
    # print("Final Neuron Output (Result is at index 0):")
    # dec_and_print(ctxt_neuron_result, keys, cc)

if __name__ == "__main__":
    main()