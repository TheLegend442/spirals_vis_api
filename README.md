
# Spiral compression analysis visualization API
This is a API used as a backend in spiral compression visualization webapp available [here](https://thelegend442.github.io/drawing_spirals/). It runs latest transformer model developed in my [diploma thesis](https://github.com/TheLegend442/spiral_compression), intakes the drawn spiral and returns compression analysis visualization.

## How to use the API
After setting up the environment from the <i>'requirements.txt'</i> file run the following command:
<pre lang="markdown">uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1</pre>
This starts the FastAPI endpoint on port 8000. To be able to use the API from another network you set up CloudFare tunnel in another terminal by running:
<pre lang="markdown">cloudflared tunnel --url http://127.0.0.1:8000</pre>
Then copy the CloudFare link and paste it to the webpage.