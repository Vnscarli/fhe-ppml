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

    # Rotations:
    # [1, 2, 4] -> Sum Tree
    # [7, 14, -4, -8] -> Repacking L1 -> L2
    context.EvalAtIndexKeyGen(keys.secretKey, [1, 2, 4, 7, 14, -4, -8])

    return keys

def get_data_weight():
    data = [0.5, 0.8, -0.2, 1.1, 3.0, 4.6, 0.6, 1]

    # Matrix of weights for 3 neurons (each with 8 weights)
    l1_weights = [
        [1.0, -1.0, 0.5, 2.0, 1.0, 1.5, 2.0, 1.0],  # Neuron 0
        [0.5,  0.5, 1.0, 1.0, 0.0, 2.0, 1.5, 0.5],  # Neuron 1
        [-1.0, 2.0, 0.0, 1.0, 1.0, 1.0, -0.5, 1.0]  # Neuron 2
    ]

    l2_weights = [
        [0.5, -0.2, 0.8, 0.0],  # Neuron 0 
        [-0.5, 0.5, 1.2, 0.0],  # Neuron 1
        [1.0, -1.0, 0.5, 0.0]   # Neuron 2 
    ]
    
    return data, l1_weights, l2_weights

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

        ctxt = ctxt_in_func     

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

def pack_data_for_layer(data, num_neurons):
    input_len = len(data)
    
    # Pack data
    packed_data = []
    for _ in range(num_neurons):
        packed_data.extend(data)
        
    return packed_data

def pack_weights_for_layer(weights_matrix):
    # Pack weights to turn into 1D vector 
    packed_weights = []
    for neuron_weights in weights_matrix:
        packed_weights.extend(neuron_weights)
        
    return packed_weights

