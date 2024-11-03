from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/process_data/", methods=["POST"])
def process_data():
    # Receive the POST request and parse JSON data
    data = request.get_json()

    print(data)
    return data


@app.route("/process_data", methods=["OPTIONS"])  # Handle CORS preflight requests
def handle_options():
    response = app.make_default_options_response()
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173/"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST"
    return response

if __name__ == "__main__":
    app.run(debug=True)
