import sys
sys.path.append('/usr/local')

import openfhe

def create_cripto_context(depth, batch_size):
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
    weights = [1.0, -1.0, 0.5, 2.0, 1.0, 1.5, 2.0, 1]
    
    return data, weights

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

def eval_single_neuron(context, ctxt_data, ctxt_weights, vector_size):
    # Dot product
    ctxt_mult = context.EvalMult(ctxt_data, ctxt_weights)

    # Sum Tree
    ctxt_sum_tree = eval_sum_tree(context, ctxt_mult, vector_size)

    # Approximation of Sigmpid Function (-0.004*x^3 + 0.197*x + 0.5)
    coefficients = [0.5, 0.197, 0.0, -0.004]

    # Evaluate polynomial homomorphically
    ctxt_activation = context.EvalPoly(ctxt_sum_tree, coefficients)
    
    return ctxt_activation

def pack_data_for_layer(data, weights_matrix, num_neurons):
    input_len = len(data)
    pass

def main():
    # Instantiate the cryptographic context
    cc = create_cripto_context(depth=4, batch_size=8)

    # Generate keys (public, private, and rotation/multiplication keys)
    keys = gen_keys(cc)

    # Load inputs and weights
    data, weights = get_data_weight()

    # Encrypt vectors
    ctxt_data = enc(data, cc, keys)
    ctxt_weights = enc(weights, cc, keys)

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
    print("Executing Single Neuron (Dot Product + Sigmoid Approximation)...")
    ctxt_neuron_result = eval_single_neuron(cc, ctxt_data, ctxt_weights, len(data))
    
    print("Final Neuron Output (Result is at index 0):")
    dec_and_print(ctxt_neuron_result, keys, cc)

if __name__ == "__main__":
    main()