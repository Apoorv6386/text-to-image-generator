from diffusers import StableDiffusionPipeline
import torch
import gradio as gr

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_auth_token=True  # Secure way to authenticate in Hugging Face Spaces
).to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Image(label="Generated Image"),
    title="Text-to-Image Generator",
    description="Enter a prompt to generate an image using Stable Diffusion"
).launch()