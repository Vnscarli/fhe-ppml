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

def data_weight():
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

def main():
    pass

if __name__ == "__main__":
    main()