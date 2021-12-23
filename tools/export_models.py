import argparse
import torch

# Please note: Add the Monodepth2 repository path to the PYTHONPATH environment variable
import networks


def convert(input_encoder, input_decoder, output_encoder, output_decoder, device):
    # Select device (cpu, cuda)
    device = torch.device(device)

    # Loading pretrained encoder model
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(input_encoder, map_location=device)
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    # Loading pretrained decoder model
    decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict_dec = torch.load(input_decoder, map_location=device)
    decoder.load_state_dict(loaded_dict_dec)
    decoder.to(device)
    decoder.eval()

    with torch.no_grad():
        # Save encoder torchscript model
        example_input = torch.rand(1, 3, feed_width, feed_height) if device == 'cpu' else torch.rand(1, 3, feed_width, feed_height).cuda()
        traced_script_module_enc = torch.jit.trace(encoder, example_input, strict=False)
        traced_script_module_enc.save(output_encoder)
        # Save decoder torchscript model
        example_features = encoder(example_input)
        # Please note: To avoid errors the decoder in the Monodepth2 repository needs to be adjusted to return the last tuple element (self.outputs[("disp",0)])
        traced_script_module_dec = torch.jit.trace(decoder, (example_features,), strict=True)
        traced_script_module_dec.save(output_decoder)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Script to convert pretrained Monodepth2 encoder/decoder models to torchscript models.')
    parser.add_argument('--input_encoder_path', type=str, help='Pretrained encoder model path')
    parser.add_argument('--input_decoder_path', type=str, help='Pretrained decoder model path')
    parser.add_argument('--output_encoder_path', type=str, help='Torchscript encoder model path')
    parser.add_argument('--output_decoder_path', type=str, help='Torchscript decoder model path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Hardware device (cpu, cuda)')
    args = parser.parse_args()
    
    # Convert pretrained encoder/decoder models to torchscript models
    convert(args.input_encoder_path, args.input_decoder_path, args.output_encoder_path, args.output_decoder_path, args.device)
