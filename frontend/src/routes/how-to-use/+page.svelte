<svelte:head>
    <title>How to use this tool</title>
    <meta name="description" content="How to use this tool" />
</svelte:head>

<div class="text-column">
    <h1>How to use this circuit solving tool</h1>
    <hr class="section-divider">
	<h3>
		Terms:
	</h3>
	<br>
	Node: A component in the circuit i.e. could be a lamp, a TV, etc. <br>
	Current: Represents the net flow of current leaving a component <br>
	Voltage: A potential difference between two parts of the circuit i.e. the energy going across a component <br>
	Edge: A connection between two components of a circuit i.e. a wire or a resistor <br>
	Conductance: A measure of how difficult it is for current to travel across an edge, also conductance = 1 / resistance <br>
	Source: The component which has a net positive current i.e. current comes out of this e.g. a battery <br>
	Sink: The component which has a net negative current i.e. the grounded node or where the current flows in to <br>
	<hr class="section-divider">
	<h3>
		Instructions:
	</h3>
    <p>
        Choose an initial current that will be fed into the source node. This program allows the user to change the current 
		instead of the voltage as it means only one value needs to be set.
    </p>
    <p>
        You can then add extra nodes (components) using the add node button. Move these nodes by dragging them with your mouse.
	</p>
	<p>
		Using the add edge button you can draw edges between the different nodes. All you have to do is click the add edge button and then
		click two nodes in succession. Clicking just one node and then pressing add edge will make you stop finding an edge i.e. you will have
		to click the button again and then click on two nodes to draw an edge.  <br>
		By default these wires have unit conductance (which also corresponds to unit resistance) but you can toggle the ability
		to manually set this value yourself! The conductance value is shown in red on the edge.
    </p>
    <p>
        Once your circuit is built you can solve the circuit by using the solve circuit button which will fill in the voltage values
		of each component that are by default "x". Note that this is only possible if there exists a path between nodes 1 and 2
		although the program should work out whether there is a path so it will prompt you if there is a problem!
    </p>
	<hr class="section-divider">
	<h3>
		Things of note:
	</h3>
	<p>
		All nodes except the initial two nodes have zero current which means that while there is current passing through them, the net
		current coming out is 0. For this reason they are called Kirchoff's Current Law nodes, or KCL nodes, as his law is:
		<h4>The algebraic sum of all currents entering and exiting a node must equal zero.</h4>
		The other nodes are called boundary nodes as they have a non-zero current loss. 
	<p>
		We use conductance instead of resistance as it makes the maths neater. It is very simply 1/resistance though. The units are
		Ω^−1 (ohms^-1) or S (Siemens).
	</p>
</div>

<style>
    .section-divider {
        border: 0; 
        border-top: 1px solid black; 
        margin: 20px 0; 
    }
</style>
