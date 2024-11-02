from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/process_data/", methods=["POST"])
def process_data():
    # Receive the POST request and parse JSON data
    data = request.get_json()

    # Extract the edges dictionary
    edges = data.get("edges")
    f1 = data.get("current1")

    # calculate the voltages given the edge layout and the source current
    if edges:
        voltages = main(edges, f1)
    else:
        voltages = []

    return jsonify(list(voltages))


@app.route("/process_data", methods=["OPTIONS"])  # Handle CORS preflight requests
def handle_options():
    response = app.make_default_options_response()
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173/"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST"
    return response

if __name__ == "__main__":
    app.run(debug=True)
