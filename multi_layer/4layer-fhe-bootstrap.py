import sys
import time
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

    levels_for_network = 9

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

    
    cc.EvalBootstrapSetup(level_budget, [0,0], batch_size)

    return cc

def gen_keys(context, batch_size):
    keys = context.KeyGen()
    context.EvalMultKeyGen(keys.secretKey)

    # Rotations:
    # [1, 2, 4] -> Sum Tree
    # [3, 6, 7, 14, -4, -8] -> Repacking L1 -> L2
    context.EvalAtIndexKeyGen(keys.secretKey, [1, 2, 3, 4, 6, 7, 14, -4, -8])

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
    l3_weights = [
        [0.1, -0.2, 0.8, 0.0],  # Neuron 0 
        [-0.5, 0.5, 1.2, 0.0],  # Neuron 1
        [1.0, -0.3, 0.5, 0.0]   # Neuron 2 
    ]
    l4_weights = [
        [0.5, -0.4, 0.8, 0.0],  # Neuron 0 
        [-0.1, 0.5, 1.2, 0.0],  # Neuron 1
        [1.0, -1.0, 0.7, 0.0]   # Neuron 2 
    ]
    
    return data, l1_weights, l2_weights, l3_weights, l4_weights

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

def repack_layer(context, ctxt_in, in_block, in_neurons, out_block, out_neurons, batch_size):
    #Takes outputs from layer one and reorganize on layer 2
    ctxt_dense = None
    
    for i in range(in_neurons):
        shift = (i * in_block) - i
        
        ctxt_shifted = context.EvalAtIndex(ctxt_in, shift) if shift > 0 else ctxt_in
        
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
    ctxt_packed = ctxt_dense
    for n in range(1, out_neurons):
        shift_right = -1 * out_block * n
        ctxt_dup = context.EvalAtIndex(ctxt_dense, shift_right)
        ctxt_packed = context.EvalAdd(ctxt_packed, ctxt_dup)
        
    return ctxt_packed

def calculate_cleartext_expected(data, weights_matrix):
    expected_results = []
    for weights in weights_matrix:
        dot_product = sum(d * w for d, w in zip(data, weights))
        activation = -0.004 * (dot_product ** 3) + 0.197 * dot_product + 0.5
        expected_results.append((dot_product, activation))
    return expected_results

def write_benchmark_report(filename, num_iterations, metrics):
    avg_boot1 = sum(metrics["boot_1"]) / num_iterations
    avg_boot2 = sum(metrics["boot_2"]) / num_iterations
    avg_boot3 = sum(metrics["boot_3"]) / num_iterations
    total_boot_time = avg_boot1 + avg_boot2 + avg_boot3
    
    avg_noise_0 = sum(metrics["noise_n0"]) / num_iterations
    avg_noise_1 = sum(metrics["noise_n1"]) / num_iterations
    avg_noise_2 = sum(metrics["noise_n2"]) / num_iterations
    total_avg_noise = (avg_noise_0 + avg_noise_1 + avg_noise_2) / 3

    with open(filename, "w") as f:
        f.write("====================================================\n")
        f.write("      BENCHMARK - FHE NEURAL NETWORK (4 LAYERS)     \n")
        f.write("====================================================\n\n")
        f.write(f"Total iterations executed: {num_iterations}\n\n")
        
        f.write("--- AVERAGE BOOTSTRAPPING TIMES ---\n")
        f.write(f"Bootstrap 1 (After L1): {avg_boot1:.4f} seconds\n")
        f.write(f"Bootstrap 2 (After L2): {avg_boot2:.4f} seconds\n")
        f.write(f"Bootstrap 3 (After L3): {avg_boot3:.4f} seconds\n")
        f.write(f"-> TOTAL AVERAGE BOOTSTRAP TIME PER INFERENCE: {total_boot_time:.4f} seconds\n\n")
        
        f.write("--- FINAL AVERAGE NOISE (Absolute Error in Layer 4) ---\n")
        f.write(f"Neuron 0: {avg_noise_0}\n")
        f.write(f"Neuron 1: {avg_noise_1}\n")
        f.write(f"Neuron 2: {avg_noise_2}\n")
        f.write(f"-> OVERALL NETWORK AVERAGE NOISE: {total_avg_noise}\n")

def main():
    BATCH_SIZE = 32
    NUM_ITERATIONS = 2
    OUTPUT_FILE = "benchmark_results.txt"

    cc = create_crypto_context(batch_size=BATCH_SIZE)
    keys = gen_keys(cc, batch_size=BATCH_SIZE)
    

    data, l1_w, l2_w, l3_w, l4_w = get_data_weight()


    # Plain text operation
    expected_l1 = calculate_cleartext_expected(data, l1_w) 
    out_l1 = [res[1] for res in expected_l1] + [0.0]

    expected_l2 = calculate_cleartext_expected(out_l1, l2_w)
    out_l2 = [res[1] for res in expected_l2] + [0.0]

    expected_l3 = calculate_cleartext_expected(out_l2, l3_w)
    out_l3 = [res[1] for res in expected_l3] + [0.0]

    expected_l4 = calculate_cleartext_expected(out_l3, l4_w)

    # Dictionary to store metrics for all iterations
    metrics = {
        "boot_1": [], "boot_2": [], "boot_3": [],
        "noise_n0": [], "noise_n1": [], "noise_n2": []
    }

    # Inference Loop

    for it in range(1, NUM_ITERATIONS + 1):
        print(f"\n==========================================")
        print(f"       STARTING ITERATION {it}/{NUM_ITERATIONS}")
        print(f"==========================================")

        # Encrypting weights inside the loop 
        ctxt_w1 = enc(pack_weights_for_layer(l1_w), cc, keys)
        ctxt_w2 = enc(pack_weights_for_layer(l2_w), cc, keys)
        ctxt_w3 = enc(pack_weights_for_layer(l3_w), cc, keys)
        ctxt_w4 = enc(pack_weights_for_layer(l4_w), cc, keys)

        # Encrypt data
        p_data = pack_data_for_layer(data, len(l1_w))
        ctxt_out = enc(p_data, cc, keys)

        # Layer 1 -> Repack -> Bootstrap 1
        print("Executing Layer 1 and Repacking...")
        ctxt_out = eval_layer(cc, ctxt_out, ctxt_w1, block_size=8)
        ctxt_out = repack_layer(cc, ctxt_out, 8, 3, 4, 3, BATCH_SIZE)
        
        print("Starting Bootstrapping 1...")
        t0 = time.time()
        ctxt_out = cc.EvalBootstrap(ctxt_out)
        t1 = time.time()
        metrics["boot_1"].append(t1 - t0)

        # Layer 2 -> Repack -> Bootstrap 2
        print("Executing Layer 2 and Repacking...")
        ctxt_out = eval_layer(cc, ctxt_out, ctxt_w2, block_size=4)
        ctxt_out = repack_layer(cc, ctxt_out, 4, 3, 4, 3, BATCH_SIZE)

        print("Starting Bootstrapping 2...")
        t0 = time.time()
        ctxt_out = cc.EvalBootstrap(ctxt_out)
        t1 = time.time()
        metrics["boot_2"].append(t1 - t0)

        # Layer 3 -> Repack -> Bootstrap 3
        print("Executing Layer 3 and Repacking...")
        ctxt_out = eval_layer(cc, ctxt_out, ctxt_w3, block_size=4)
        ctxt_out = repack_layer(cc, ctxt_out, 4, 3, 4, 3, BATCH_SIZE) # L3 -> L4 (3 neurons)

        print("Starting Bootstrapping 3...")
        t0 = time.time()
        ctxt_out = cc.EvalBootstrap(ctxt_out)
        t1 = time.time()
        metrics["boot_3"].append(t1 - t0)

        # Layer 4 (Final)
        print("Executing Layer 4...")
        ctxt_out = eval_layer(cc, ctxt_out, ctxt_w4, block_size=4)

        # Decrypt and Track Noise
        ptxt_res = cc.Decrypt(ctxt_out, keys.secretKey)
        ptxt_res.SetLength(len(l4_w) * 4)
        res_array = ptxt_res.GetRealPackedValue()

        for i in range(len(l4_w)):
            valid_index = i * 4 
            val_fhe = res_array[valid_index]
            dot_clear, val_clear = expected_l4[i]
            error = abs(val_fhe - val_clear)
            
            # Store noise in metrics
            metrics[f"noise_n{i}"].append(error)

            print(f"Neuron {i}:")
            print(f"Raw dot product: {dot_clear}")
            print(f"Expected Output: {val_clear}")
            print(f"FHE Output: {val_fhe}")
            print(f"Absolute error (noise): {error}\n")
            
        print(f"Iteration {it} finished!")
        # Outpu metrics in file
    write_benchmark_report(OUTPUT_FILE, NUM_ITERATIONS, metrics)
    print(f"\n[SUCCESS] Benchmark complete! Results saved in '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()