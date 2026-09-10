import sys
sys.path.append('/usr/local')

import openfhe

def dec_and_print(ctxt, keys, openFheContext):
    res = openFheContext.Decrypt(ctxt, keys.secretKey)
    print(res)

def main():
    parameters = openfhe.CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(5) # Number of allowed multiplications
    parameters.SetScalingModSize(50)  # Scale size (in bits, e.g., 2^50)
    parameters.SetBatchSize(8)

    cc = openfhe.GenCryptoContext(parameters)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.ADVANCEDSHE)

    # Generate keys
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey) # Keys for multiplication
    cc.EvalAtIndexKeyGen(keys.secretKey, [1, 2, 4, 8]) # Keys for rotation


    data = [0.5, 0.8, -0.2, 1.1, 3.0, 4.6, 0.6, 1]
    weights = [1.0, -1.0, 0.5, 2.0, 1.0, 1.5, 2.0, 1] 
    
        
    
    ptxt_data = cc.MakeCKKSPackedPlaintext(data) # Encode as plaintext
    ctxt_data = cc.Encrypt(keys.publicKey, ptxt_data) # Encrypt into ciphertext

    ptxt_weights = cc.MakeCKKSPackedPlaintext(weights) # Encode as plaintext

    ctxt_mult = cc.EvalMult(ctxt_data, ptxt_weights) # Multiply cipher x plain
    
    

    dec_and_print(ctxt_mult, keys, cc)

    ctxt_rot1 = cc.EvalRotate(ctxt_mult, 1)
    ctxt_sum1 = cc.EvalAdd(ctxt_mult, ctxt_rot1)
    
    ctxt_rot2 = cc.EvalRotate(ctxt_sum1, 2)
    ctxt_sum2 = cc.EvalAdd(ctxt_sum1, ctxt_rot2)
    

    ctxt_rot3 = cc.EvalRotate(ctxt_sum2, 4)
    ctxt_sum3 = cc.EvalAdd(ctxt_sum2, ctxt_rot3)

    multsum = sum(d * p for d, p in zip(data, weights))
    print(f"Dot product sum: {multsum:.3f}")

    dec_and_print(ctxt_sum3, keys, cc)
    
    coefficients = [0.1, 0.5, 1.0] 

    expected_res = coefficients[0] + (coefficients[1] * multsum) + (coefficients[2] * (multsum ** 2))
    print(f"Expected result after polynomial: {expected_res:.3f}")

    ctxt_activation = cc.EvalPoly(ctxt_sum3, coefficients)

    res = cc.Decrypt(ctxt_activation, keys.secretKey)

    
    print(res)

if __name__ == "__main__":
    main()