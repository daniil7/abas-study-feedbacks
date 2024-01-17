<script>
    export let text;
</script>

<button on:click
    {...$$restProps} >
    {text}
</button>

<style>
    button {
        border-radius: 0.25rem;
        padding: 0.25rem, 0.5rem, 0.25rem, 0.5rem;
        background-color: white;
        color: black;
        border: 2px solid black;
        cursor: pointer;
    }
</style>
