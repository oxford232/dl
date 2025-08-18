import sentencepiece as spm

sp_en = spm.SentencePieceProcessor()
sp_en.load('en_bpe.model')

sp_cn = spm.SentencePieceProcessor()
sp_cn.load('zh_bpe.model')

text = 'The weather is good today.'
text2 = "今天天气非常好。"

encode_result = sp_en.encode(text, out_type = int)
encode_result2 = sp_cn.encode(text2, out_type = int)

print("encode: ", encode_result, encode_result2)

decode_result = sp_en.decode(encode_result)
decode_result2 = sp_cn.decode(encode_result2)

print("decode: ", decode_result, decode_result2)
