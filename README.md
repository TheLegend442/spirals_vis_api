
# Spiral compression analysis visualization API
This is a API used as a backend in spiral compression visualization webapp available [here](https://thelegend442.github.io/drawing_spirals/). It runs latest transformer model developed in my [diploma thesis](https://github.com/TheLegend442/spiral_compression), intakes the drawn spiral and returns compression analysis visualization.

## Repository content:
<ul>
<li><i>chronos_emb.py</i> - helper script to perform spiral embedding using Chronos-2 model.</li>
<li><i>transformer.py</i> - script containing transformer model definition, dataset definition and training process. It is used to train the model with specified parameters on specified dataset.</li>
<li><i>inference.py</i> - script that performs the inference with specified model and saves some basic plots. It can also calculate loss on a test set.</li>
<li><i>server.py</i> - implementation of FastAPI - it intakes a spiral <i>'.csv'</i> file and return png with the results
<li><i>helper_functions.py</i> - helper script containing some functions common to multiple scripts to avoid mismatching versions.</li>
<li><i>index.html</i> - HTML script of the webapp.
</ul>

## How to use the API
After setting up the environment from the <i>'requirements.txt'</i> file run the following command:
<pre lang="markdown">uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1</pre>
This starts the FastAPI endpoint on port 8000. To be able to use the API from another network you set up CloudFare tunnel in another terminal by running:
<pre lang="markdown">cloudflared tunnel --url http://127.0.0.1:8000</pre>
Then copy the CloudFare link and paste it to the webpage.

### Note:
The code in this repository is an assembly of scripts that were used to research and write my diploma thesis.<br>