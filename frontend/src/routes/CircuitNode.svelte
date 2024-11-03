<!-- CircuitNode.svelte -->
<script>
    import { createEventDispatcher } from "svelte";

    export let x = 100; // Initial x-coordinate
    export let y = 100; // Initial y-coordinate
    
    const dispatch = createEventDispatcher();

    let isDragging = false;

    function onMouseDown() {
        isDragging = true;
    }

    function onMouseMove(e) {
        if (isDragging) {
            x += e.movementX;
            y += e.movementY;
        }
    }

    function onMouseUp() {
        isDragging = false;
    }

    function handleClick() {
        dispatch('nodeClick', { x, y });
    }
</script>

<main>
    <button class="node" tabindex="0" on:mousedown={onMouseDown} on:click={handleClick}
        style="left: {x}px; top: {y}px;">
        <span>
            <slot name="CircleText" header=1 voltage=1 current=0/>
        </span>
    </button>
</main>

<svelte:window on:mouseup={onMouseUp} on:mousemove={onMouseMove} />


<style>
    main {
        margin: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 2vh;
    }

    .node {
        width: 100px;
        height: 100px;
        border: none;
        border-radius: 50%;
        background-color: #3498db;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.2);
        transition: box-shadow 0.3s ease;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column; /* Stacks the child elements vertically */

        position: absolute;
        cursor: grab;
    }

    .node span {
        text-align: center;
        line-height: 1;
        font-size: 12px; /* Adjust the font size as needed */
        color: white;
        padding: 5px;
    }

        .node:hover {
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.4);
    }


</style>