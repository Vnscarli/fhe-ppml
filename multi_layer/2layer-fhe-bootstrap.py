import sys
sys.path.append('/usr/local')

import openfhe

def create_crypto_context(batch_size):
    parameters = openfhe.CCParamsCKKSRNS()
    # Ternary key for bootstrapping
    secret_key_dist = openfhe.SecretKeyDist.UNIFORM_TERNARY
    parameters.SetSecretKeyDist(secret_key_dist)
    parameters.SetSecurityLevel(openfhe.SecurityLevel.HEStd_NotSet)
    parameters.SetRingDim(1 << 15)

    level_budget = [4, 4]

    levels_for_network = 7

    bootstrap_depth = openfhe.FHECKKSRNS.GetBootstrapDepth(level_budget, secret_key_dist)
    depth = levels_for_network + bootstrap_depth

    parameters.SetMultiplicativeDepth(depth)
    parameters.SetFirstModSize(60)
    parameters.SetScalingModSize(50)
    parameters.SetBatchSize(batch_size)

    cc = openfhe.GenCryptoContext(parameters)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.ADVANCEDSHE)
    # Needs FHE feature for Bootstrapping
    cc.Enable(openfhe.PKESchemeFeature.FHE)

    # Level to be consumed at bootstrapping
    
    cc.EvalBootstrapSetup(level_budget, [0,0], batch_size)

    return cc

def gen_keys(context, batch_size):
    keys = context.KeyGen()
    context.EvalMultKeyGen(keys.secretKey)

    # Rotations:
    # [1, 2, 4] -> Sum Tree
    # [7, 14, -4, -8] -> Repacking L1 -> L2
    context.EvalAtIndexKeyGen(keys.secretKey, [1, 2, 4, 7, 14, -4, -8])

    # Bootstrap keys
    context.EvalBootstrapKeyGen(keys.secretKey, batch_size)

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

def repack_l1_to_l2(context, ctxt_l1_out, l1_block, num_l1_neurons, l2_block, num_l2_neurons, batch_size):
    #Takes outputs from layer one and reorganize on layer 2
    ctxt_dense = None
    
    for i in range(num_l1_neurons):
        shift = (i * l1_block) - i
        
        ctxt_shifted = context.EvalAtIndex(ctxt_l1_out, shift) if shift > 0 else ctxt_l1_out
        
        # Create a mask to clear surrounding noise
        mask = [0.0] * batch_size
        mask[i] = 1.0
        ptxt_mask = context.MakeCKKSPackedPlaintext(mask)
        
        ctxt_masked = context.EvalMult(ctxt_shifted, ptxt_mask)
        
        if ctxt_dense is None:
            ctxt_dense = ctxt_masked
        else:
            ctxt_dense = context.EvalAdd(ctxt_dense, ctxt_masked)
            
    # Duplicate the clean block for each neuron at layer 2
    ctxt_packed_l2 = ctxt_dense
    for n in range(1, num_l2_neurons):
        shift_right = -1 * l2_block * n
        ctxt_dup = context.EvalAtIndex(ctxt_dense, shift_right)
        ctxt_packed_l2 = context.EvalAdd(ctxt_packed_l2, ctxt_dup)
        
    return ctxt_packed_l2

def calculate_cleartext_expected(data, weights_matrix):
    expected_results = []
    for weights in weights_matrix:
        dot_product = sum(d * w for d, w in zip(data, weights))
        activation = -0.004 * (dot_product ** 3) + 0.197 * dot_product + 0.5
        expected_results.append((dot_product, activation))
    return expected_results

def main():
    BATCH_SIZE = 32
    cc = create_crypto_context(batch_size=BATCH_SIZE)
    keys = gen_keys(cc, batch_size=BATCH_SIZE)

    data, l1_weights, l2_weights = get_data_weight()


    # Clear text operation
    l1_expected = calculate_cleartext_expected(data, l1_weights) 

    l1_activations = [res[1] for res in l1_expected]
    l1_activations.append(0.0)

    l2_expected = calculate_cleartext_expected(l1_activations, l2_weights)

    # Layer 1
    print("Executing Layer 1")
    num_l1_neurons = len(l1_weights)

    p_data_l1 = pack_data_for_layer(data, num_l1_neurons)
    p_weights_l1 = pack_weights_for_layer(l1_weights)

    ctxt_data_l1 = enc(p_data_l1, cc, keys)
    ctxt_weights_l1 = enc(p_weights_l1, cc, keys)

    ctxt_l1_out = eval_layer(cc, ctxt_data_l1, ctxt_weights_l1, block_size = 8)

    # Repack (L1 -> L2)
    print("Repacking encypted data for layer 2")
    ctxt_data_l2 = repack_l1_to_l2(cc, ctxt_l1_out, 8, 3, 4, 3, BATCH_SIZE)

    # Bootstrapping
    print(f"Starting Bootstrapping")
    ctxt_data_l2_refreshed = cc.EvalBootstrap(ctxt_data_l2)
    print("Bootstrap finished!\n")

    # Layer 2
    print("Executing Layer 2")

    p_weights_l2 = pack_weights_for_layer(l2_weights) 
    ctxt_weights_l2 = enc(p_weights_l2, cc, keys)

    ctxt_l2_out = eval_layer(cc, ctxt_data_l2, ctxt_weights_l2, block_size = 4)

    # Decrypt and Compare
    ptxt_res = cc.Decrypt(ctxt_l2_out, keys.secretKey)

    ptxt_res.SetLength(len(l2_weights)*4)
    res_array = ptxt_res.GetRealPackedValue()

    for i in range(len(l2_weights)):
        valid_index = i * 4 
        val_fhe = res_array[valid_index]

        dot_clear, val_clear = l2_expected[i]
        error = abs(val_fhe - val_clear)

        print(f"Neuron {i}:")
        print(f"Raw dot product: {dot_clear:.4f}")
        print(f"Expected Output: {val_clear:.8f}")
        print(f"FHE Output: {val_fhe:.8f}")
        print(f"Absolute error (noise): {error}")



if __name__ == "__main__":
    main()