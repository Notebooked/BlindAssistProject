from gradio_client import Client, handle_file

client = Client("leomcd/BlindCustomYolo")
result = client.predict(
		image=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
		api_name="/predict"
)
print(result)