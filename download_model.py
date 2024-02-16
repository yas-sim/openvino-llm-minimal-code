model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
#model_id = 'Intel/neural-chat-7b-v3'
model_vendor, model_name = model_id.split('/')

FP16_GEN, INT8_GEN, INT4_GEN = True, True, True

if FP16_GEN:
    from optimum.intel.openvino import OVModelForCausalLM
    ov_model=OVModelForCausalLM.from_pretrained(model_id=model_id, export=True, compile=False, load_in_8bit=False)
    ov_model.half()
    ov_model.save_pretrained(f'{model_name}/FP16')

if INT8_GEN:
    from optimum.intel.openvino import OVModelForCausalLM
    from optimum.intel import OVQuantizer
    ov_model=OVModelForCausalLM.from_pretrained(model_id=model_id, export=True, compile=False, load_in_8bit=False)
    quantizer = OVQuantizer.from_pretrained(ov_model)
    quantizer.quantize(save_directory=f'{model_name}/INT8', weights_only=True)

if INT4_GEN:
    import shutil
    from optimum.intel.openvino import OVModelForCausalLM
    import openvino as ov
    import nncf
    ov_model=OVModelForCausalLM.from_pretrained(model_id=model_id, export=True, compile=False, load_in_8bit=False)
    compressed_model = nncf.compress_weights(ov_model.half()._original_model, mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128, ratio=0.8)
    ov.save_model(compressed_model, f'{model_name}/INT4/openvino_model.xml')
    shutil.copy(f'{model_name}/FP16/config.json', f'{model_name}/INT4/config.json')
