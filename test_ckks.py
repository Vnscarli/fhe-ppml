import sys
sys.path.append('/usr/local')

import openfhe

def dec_and_print(ctxt, keys, openFheContext):
    resultado = openFheContext.Decrypt(ctxt, keys.secretKey)
    print(resultado)

def main():
    parameters = openfhe.CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(5) # Qt de multiplicações acceitas
    parameters.SetScalingModSize(50) # Escala (em bits 2^50) que demonstra
    parameters.SetBatchSize(8)

    cc = openfhe.GenCryptoContext(parameters)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(openfhe.PKESchemeFeature.ADVANCEDSHE)

    # Gerar chaves 
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey) # Para multiplicação
    cc.EvalAtIndexKeyGen(keys.secretKey, [1, 2, 4, 8]) # Chaves para rotação


    dados = [0.5, 0.8, -0.2, 1.1, 3.0, 4.6, 0.6, 1]
    pesos = [1.0, -1.0, 0.5, 2.0, 1.0, 1.5, 2.0, 1] 
    
        
    
    ptxt_dados = cc.MakeCKKSPackedPlaintext(dados) # Transforma em plaintext
    ctxt_dados = cc.Encrypt(keys.publicKey, ptxt_dados) # Transforma em ciphertext

    ptxt_pesos = cc.MakeCKKSPackedPlaintext(pesos) # Transforma em palintext

    ctxt_mult = cc.EvalMult(ctxt_dados, ptxt_pesos) # Multiplica plain x cipher
    
    

    dec_and_print(ctxt_mult, keys, cc)

    ctxt_rot1 = cc.EvalRotate(ctxt_mult, 1)
    ctxt_soma1 = cc.EvalAdd(ctxt_mult, ctxt_rot1)
    
    ctxt_rot2 = cc.EvalRotate(ctxt_soma1, 2)
    ctxt_soma2 = cc.EvalAdd(ctxt_soma1, ctxt_rot2)
    

    ctxt_rot3 = cc.EvalRotate(ctxt_soma2, 4)
    ctxt_soma3 = cc.EvalAdd(ctxt_soma2, ctxt_rot3)

    multsum = sum(d * p for d, p in zip(dados, pesos))
    print(f"Soma das multiplicações: {multsum:.3f}")

    dec_and_print(ctxt_soma3, keys, cc)
    
    coeficientes = [0.1, 0.5, 1.0] 

    resultado_esperado = coeficientes[0] + (coeficientes[1] * multsum) + (coeficientes[2] * (multsum ** 2))
    print(f"Resultado esperado após polinômio: {resultado_esperado:.3f}")

    ctxt_ativacao = cc.EvalPoly(ctxt_soma3, coeficientes)

    resultado = cc.Decrypt(ctxt_ativacao, keys.secretKey)

    
    print(resultado)

if __name__ == "__main__":
    main()