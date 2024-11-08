<script>
	let space_ready = false;
	let highest_region_id = 0;
	let regions = {};
	let region_costs = {};
	let isAssigningCosts = false;
	let isBinHighlighted = false;
	let highlighted_method = null;

	const methods = [
		{"id": "ray", "label": "Ray Pathfinding"},
		{"id": "djikstra", "label": "Djikstra"},
		{"id": "astar", "label": "A*"},
		{"id": "bidirectional", "label": "Bi-directional Search"}
	]
  
	// Drawing state
	let isDrawing = false;
	let startPoint = null;
	let points = [];
	let lines = [];
	let currentLine = null;
	let canvas;
	let ctx;
	let rect;
	let isValidLine = true;
	let closed_regions = [];
	let closed_regions_edges = [];
	
	// Start and end points
	let startEndPoints = {
	  start: null,
	  end: null
	};

	function POST_data(id) {
		const backendUrl = "http://127.0.0.1:5000";

		// Format data as an array of objects with `id` and `points`

		console.log(closed_regions_edges);

		const dataToSend = closed_regions_edges.map((region, index) => ({
			id: index,
			edges: region.edges,
			cost: region_costs[index] !== undefined ? region_costs[index] : 1
		}));

		console.log(dataToSend);

		fetch(backendUrl + "/process_data/", {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({"data": dataToSend, "target": { x: canvas.width - 5, y: canvas.height - 5 }, "method": id})
		})
		.then(response => {
			if (response.ok) {
				console.log("Data sent to backend successfully");
			} else {
				console.error("Failed to send data to backend");
			}
		})
		.then(returned_data => {
			console.log(returned_data);
			
		})
		.catch(error => {
			console.error("Error sending data to backend:", error);
		});
	}

	function handleBinClick() {
		isBinHighlighted = true;
		lines = [];
		points = [];
		regions = {};
		region_costs = {};
		closed_regions = [];
		closed_regions_edges = [];
		highest_region_id = 0;
		isAssigningCosts = false;
		initializeWalls();
		redraw();

		// Remove highlight after 200ms
		setTimeout(() => {
			isBinHighlighted = false;
			redraw();
		}, 200);
	}

	function initCanvas(element) {
	  canvas = element;
	  ctx = canvas.getContext('2d');
	  rect = canvas.getBoundingClientRect();
	  
	  canvas.width = canvas.offsetWidth;
	  canvas.height = canvas.offsetHeight;
	  
	  // Initialize wall lines and special points
	  initializeWalls();
	  redraw();
	}
  
	function initializeWalls() {
		// Create border points
		const topLeft = { x: 5, y: 5 };
		const topRight = { x: canvas.width - 5, y: 5 };
		const bottomLeft = { x: 5, y: canvas.height - 5 };
		const bottomRight = { x: canvas.width - 5, y: canvas.height - 5 };
		
		points = [topLeft, topRight, bottomLeft, bottomRight];
		
		lines = [
			{ start: topLeft, end: topRight },
			{ start: topRight, end: bottomRight },
			{ start: bottomRight, end: bottomLeft },
			{ start: bottomLeft, end: topLeft }
		];

		// Set start and end points exactly in corners
		startEndPoints.start = { x: 5, y: 5 };
		startEndPoints.end = { x: canvas.width - 5, y: canvas.height - 5 };
	}
  
	function findCycles() {
		const adjacencyList = new Map();
		
		// Create adjacency list from lines
		for (let line of lines) {
			if (!adjacencyList.has(line.start)) {
				adjacencyList.set(line.start, new Set());
			}
			if (!adjacencyList.has(line.end)) {
				adjacencyList.set(line.end, new Set());
			}
			adjacencyList.get(line.start).add(line.end);
			adjacencyList.get(line.end).add(line.start);
		}

		const cycles = new Set();
		const visited = new Set();
		const inCurrentPath = new Set();
		const allPoints = new Set(Array.from(adjacencyList.keys()));

		function getCanonicalCycle(cycle) {
			// Helper function to convert points to comparable string
			const pointToString = point => `${point.x},${point.y}`;

			// Find all possible rotations and reversals
			let allVersions = [];

			// Add all rotations
			for (let i = 0; i < cycle.length; i++) {
				const rotation = [...cycle.slice(i), ...cycle.slice(0, i)];
				allVersions.push(rotation.map(pointToString).join('|'));

				// Add reversed version of this rotation
				const reversed = [...rotation].reverse();
				allVersions.push(reversed.map(pointToString).join('|'));
			}

			// Return the lexicographically smallest version
			return allVersions.sort()[0];
		}

		function findCyclesFromNode(start, current, path = [], depth = 0) {
			inCurrentPath.add(current);
			path.push(current);

			const neighbors = adjacencyList.get(current) || new Set();
			for (let neighbor of neighbors) {
				if (neighbor === start && path.length > 2) {
					const cycle = [...path];
					const canonicalCycle = getCanonicalCycle(cycle);
					
					// Convert cycle to region format for checking
					const region = {
						points: cycle.map(point => ({
							x: Math.round(point.x),
							y: Math.round(point.y)
						}))
					};

					// Check if any point from the graph is inside this potential region
					let isValidRegion = true;
					for (let point of allPoints) {
						// Skip points that are part of the region boundary
						if (cycle.some(cyclePoint => 
							cyclePoint.x === point.x && cyclePoint.y === point.y)) {
							continue;
						}
						
						if (isPointInRegion(point, region.points)) {
							isValidRegion = false;
							break;
						}
					}

					if (isValidRegion) {
						cycles.add(canonicalCycle);
					}
				} else if (!inCurrentPath.has(neighbor) && !visited.has(neighbor)) {
					findCyclesFromNode(start, neighbor, path, depth + 1);
				}
			}

			path.pop();
			inCurrentPath.delete(current);

			if (depth === 0) {
				visited.add(start);
			}
		}

		// Start only from unvisited nodes to avoid redundant searches
		for (let point of adjacencyList.keys()) {
			if (!visited.has(point)) {
				findCyclesFromNode(point, point);
			}
		}

		let region_id = 0;
		closed_regions = [];
		closed_regions_edges = [];

		Array.from(cycles).forEach(cycleStr => {
			const points = cycleStr.split('|').map(pointStr => {
				const [x, y] = pointStr.split(',').map(Number);
				return { x, y };
			});
			
			// Create region for closed_regions
			closed_regions.push({
				id: region_id,
				points: points.map(point => ({
					x: Math.round(point.x),
					y: Math.round(point.y)
				}))
			});

			// Create edges for closed_regions_edges
			const edges = [];
			for (let i = 0; i < points.length; i++) {
				const start = points[i];
				const end = points[(i + 1) % points.length]; // Use modulo to connect last point to first
				
				// Find the matching line from the original lines array
				const matchingLine = lines.find(line => 
					(line.start.x === start.x && line.start.y === start.y && 
					line.end.x === end.x && line.end.y === end.y) ||
					(line.start.x === end.x && line.start.y === end.y && 
					line.end.x === start.x && line.end.y === start.y)
				);
				
				if (matchingLine) {
					edges.push(matchingLine);
				}
			}
			
			closed_regions_edges.push({
				id: region_id,
				edges: edges
			});

			region_id++;
		});
		console.log("Outputting closed regions");
		console.log(closed_regions_edges);
		console.log(closed_regions);

		return closed_regions;
	}

	function isPointInRegion(point, region) {
		// Input validation
		if (!region || region.length < 3) return false;

		let inside = false;
		let max_dist = 0;
		let furthest_edge = null;

		// Convert region points to edges if not already in edge format
		const edges = region.map((point, index) => [
			point,
			region[(index + 1) % region.length]
		]);

		// Find the furthest edge from the point
		for (const edge of edges) {
			const edge_mid = {
				x: (edge[0].x + edge[1].x) / 2,
				y: (edge[0].y + edge[1].y) / 2
			};
			
			const dist = Math.sqrt(
				Math.pow(point.x - edge_mid.x, 2) + 
				Math.pow(point.y - edge_mid.y, 2)
			);
			
			if (dist > max_dist) {
				max_dist = dist;
				furthest_edge = edge;
			}
		}

		function orientation(p, q, r) {
			const val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
			if (val === 0) return 0; // Collinear
			return (val > 0) ? 1 : 2; // Clockwise or Counterclockwise
		}

		function onSegment(p, q, r) {
			return (q.x <= Math.max(p.x, r.x) && q.x >= Math.min(p.x, r.x) &&
					q.y <= Math.max(p.y, r.y) && q.y >= Math.min(p.y, r.y));
		}

		function segmentsIntersect(A, B, C, D) {
			// Find the four orientations needed for the general and special cases
			const o1 = orientation(A, B, C);
			const o2 = orientation(A, B, D);
			const o3 = orientation(C, D, A);
			const o4 = orientation(C, D, B);

			// General case
			if (o1 !== o2 && o3 !== o4) return true;

			// Special cases
			if (o1 === 0 && onSegment(A, C, B)) return true;
			if (o2 === 0 && onSegment(A, D, B)) return true;
			if (o3 === 0 && onSegment(C, A, D)) return true;
			if (o4 === 0 && onSegment(C, B, D)) return true;

			return false;
		}

		// Create a ray from the point to a point far along the same line as the furthest edge
		const ray_end = {
			x: point.x + (furthest_edge[1].x - furthest_edge[0].x) * 1000,
			y: point.y + (furthest_edge[1].y - furthest_edge[0].y) * 1000
		};

		// Count intersections with ray from point to ray_end
		let count = 0;
		for (const edge of edges) {
			if (segmentsIntersect(point, ray_end, edge[0], edge[1])) {
				count++;
			}
		}

		return count % 2 === 1;
	}

	function getMousePos(event) {
	  return {
		x: event.clientX - rect.left,
		y: event.clientY - rect.top
	  };
	}

	function isInsideCanvas(pos) {
	  return pos.x >= 0 && pos.x <= canvas.width && 
			 pos.y >= 0 && pos.y <= canvas.height;
	}

	function getNearestPoint(pos) {
	  for (let point of points) {
		const dx = point.x - pos.x;
		const dy = point.y - pos.y;
		const distance = Math.sqrt(dx * dx + dy * dy);
		if (distance < 10) {
		  return point;
		}
	  }
	  return null;
	}

	function doLinesIntersect(line1Start, line1End, line2Start, line2End) {
		const a = {
			x: line1End.x - line1Start.x,
			y: line1End.y - line1Start.y
		};
		const b = {
			x: line2End.x - line2Start.x,
			y: line2End.y - line2Start.y
		};

		const cross = a.x * b.y - a.y * b.x;

		if (Math.abs(cross) < 0.0001) {
			return false;
		}

		const s = ((line2Start.x - line1Start.x) * b.y - (line2Start.y - line1Start.y) * b.x) / cross;
		const t = ((line1Start.x - line2Start.x) * a.y - (line1Start.y - line2Start.y) * a.x) / -cross;

		return s >= 0 && s <= 1 && t >= 0 && t <= 1;
	}
  
	function wouldLineIntersect(start, end) {
		for (let line of lines) {
			if (start === line.start || start === line.end || 
				end === line.start || end === line.end) {
				continue;
			}

			if (doLinesIntersect(start, end, line.start, line.end)) {
				return true;
			}
		}
		return false;
	}

	function handleAssignCosts() {
		isAssigningCosts = !isAssigningCosts;  // Toggle the state
		redraw();
	}

	function handleCanvasClick(event) {
		const pos = getMousePos(event);

		// Exit if click is outside canvas
		if (!isInsideCanvas(pos)) {
			isDrawing = false;
			startPoint = null;
			redraw();
			return;
		}

		// Handle cost assignment mode
		if (isAssigningCosts) {
			// Try to find which region was clicked
			let clickedRegion = null;
			for (let region of closed_regions) {
				if (isPointInRegion(pos, region.points)) {
					clickedRegion = region;
					break;  // Exit loop once we find the containing region
				}
			}

			// If we found a region, handle cost assignment
			if (clickedRegion) {
				const cost = prompt(`Enter cost for region ${clickedRegion.id}:`);
				if (cost !== null && !isNaN(cost)) {
					region_costs[clickedRegion.id] = parseFloat(cost);
				}
				redraw();
				console.log(closed_regions);
				return;
			}
			
			// If no region was clicked, just return
			return;
		}

		// Handle drawing mode
		const nearestPoint = getNearestPoint(pos);

		if (!isDrawing) {
			// Start drawing
			if (nearestPoint) {
				startPoint = nearestPoint;
			} else {
				startPoint = pos;
				points.push(pos);
			}
			isDrawing = true;
		} else {
			// Finish drawing current line
			let endPoint = nearestPoint || pos;

			if (!wouldLineIntersect(startPoint, endPoint)) {
				if (!nearestPoint) {
					points.push(endPoint);
				}
				lines.push({
					start: startPoint,
					end: endPoint
				});
				closed_regions = findCycles();  // Update regions
			}
			isDrawing = false;
			startPoint = null;
		}

		redraw();
	}

	function handleMouseMove(event) {
	  if (!isDrawing) return;
	  
	  const pos = getMousePos(event);
	  const nearestPoint = getNearestPoint(pos);
	  const endPoint = nearestPoint || pos;
	  
	  currentLine = {
		start: startPoint,
		end: endPoint
	  };
	  
	  isValidLine = !wouldLineIntersect(startPoint, endPoint);
	  
	  redraw();
	}
  
	function redraw() {
		if (!ctx) return;

		// Clear canvas
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		
		// Draw regions
		for (let region of closed_regions) {
			if (region.points && region.points.length > 0) {
				ctx.beginPath();
				ctx.moveTo(region.points[0].x, region.points[0].y);
				for (let i = 1; i < region.points.length; i++) {
					ctx.lineTo(region.points[i].x, region.points[i].y);
				}
				ctx.closePath();
				
				// Set fill color before filling - using a semi-transparent green
				ctx.fillStyle = 'rgba(46, 204, 113, 0.3)';  // This is a light green color
				ctx.fill();

				// Reset fillStyle for the cost text
				if (region_costs[region.id] !== undefined) {
					const centerX = region.points.reduce((sum, p) => sum + p.x, 0) / region.points.length;
					const centerY = region.points.reduce((sum, p) => sum + p.y, 0) / region.points.length;
					ctx.fillStyle = '#000';
					ctx.font = '12px Arial';
					ctx.textAlign = 'center';
					ctx.fillText(`Cost: ${region_costs[region.id]}`, centerX, centerY);
				}
			}
		}
		
		// Draw permanent lines
		ctx.strokeStyle = '#2c3e50';
		ctx.lineWidth = 2;
		for (let line of lines) {
			ctx.beginPath();
			ctx.moveTo(line.start.x, line.start.y);
			ctx.lineTo(line.end.x, line.end.y);
			ctx.stroke();
		}
		
		// Draw current line if drawing
		if (isDrawing && currentLine) {
			ctx.strokeStyle = isValidLine ? '#3498db' : '#e74c3c';
			ctx.beginPath();
			ctx.moveTo(currentLine.start.x, currentLine.start.y);
			ctx.lineTo(currentLine.end.x, currentLine.end.y);
			ctx.stroke();
		}
		
		// Draw regular points
		ctx.fillStyle = '#e74c3c';
		for (let point of points) {
			ctx.beginPath();
			ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
			ctx.fill();
		}

		// Draw start and end points
		ctx.fillStyle = '#2ecc71';
		ctx.beginPath();
		ctx.arc(startEndPoints.start.x, startEndPoints.start.y, 6, 0, Math.PI * 2);
		ctx.fill();
		
		ctx.beginPath();
		ctx.arc(startEndPoints.end.x, startEndPoints.end.y, 6, 0, Math.PI * 2);
		ctx.fill();
	}

	function run_method(id) {
		highlighted_method = id;
		POST_data(id)
		// Remove highlight after 200ms
		setTimeout(() => {
			highlighted_method = null;
			redraw();
		}, 200);
	}
  </script>
  
  <div class="container">
	<header class="banner">
	  <h1>Pathfinding</h1>
	</header>
  
	<nav class="sub-banner">
	  {#each methods as method}  
		<button
			class="method-button"
			class:highlighted={highlighted_method === method.id}
			on:click={() => run_method(method.id)}
		>
			{method.label}
		</button>
	  {/each}
	</nav>
  
	<main class="content">
		<canvas 
			class="drawing-area"
			use:initCanvas
			on:click={handleCanvasClick}
			on:mousemove={handleMouseMove}
		></canvas>

		<div class="side-controls">
			<button 
				class:highlighted={isBinHighlighted}
				on:click={handleBinClick}
			>
				Bin
			</button>
			<button 
				class:highlighted={isAssigningCosts}
				on:click={handleAssignCosts}
			>
				Assign Region Costs
			</button>
		</div>
			  
	</main>
  </div>
  
  <style>
	.container {
		display: flex;
		flex-direction: column;
		min-height: 100vh;
		width: 100%;
	}
  
	.banner {
		background-color: #2c3e50;
		color: white;
		padding: 1rem;
		text-align: center;
	}
  
	.banner h1 {
		margin: 0;
		font-size: 2rem;
	}
  
	.sub-banner {
		background-color: #34495e;
		padding: 0.5rem;
		display: flex;
		gap: 0.5rem;
		justify-content: center;
		flex-wrap: wrap;
	}
  
	button {
		padding: 0.5rem 1rem;
		border: none;
		border-radius: 4px;
		cursor: pointer;
		background-color: #95a5a6;
		color: white;
		transition: all 0.3s ease;
	}
  
	button.active {
		background-color: #3498db;
	}
  
	button.disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.mode-button {
        /* Your existing button styles */
        transition: background-color 0.15s ease;
    }

    .mode-button.highlighted {
        background-color: #e2e8f0;  /* Or any color you prefer */
    }

	button:not(.disabled):hover {
		background-color: #2980b9;
	}

	.content {
		flex: 1;
		padding: 1rem;
		display: flex;
		justify-content: center;
		align-items: center;
		background-color: #ecf0f1;
	}
  
	.drawing-area {
		width: 600px;
		height: 400px;
		background-color: white;
		border: 1px solid #bdc3c7;
	}

	.side-controls {
		position: absolute;
		left: 1rem;
		top: 50%;
		transform: translateY(-50%);
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.side-controls button {
		width: 120px;
	}

	.highlighted {
		background-color: #e74c3c !important;
		transform: scale(1.05);
		transition: all 0.2s ease;
	}

	button.highlighted:hover {
		background-color: #c0392b !important;
	}
  </style>