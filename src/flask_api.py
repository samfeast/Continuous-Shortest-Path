def get_solution(data, target, method):
    pass


from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def process_data():
    # Receive the POST request and parse JSON data
    response = request.get_json()
    data = response.get("data")
    target = response.get("target")
    method = response.get("method")
    # methods are: ray, djikstra, astar, bidirectional

    solution = get_solution()

    return solution


@app.route("/process_data", methods=["OPTIONS"])  # Handle CORS preflight requests
def handle_options():
    response = app.make_default_options_response()
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173/"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST"
    return response

if __name__ == "__main__":
    app.run(debug=True)
