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
        border: 1px solid black;
        cursor: pointer;
    }
    @media (prefers-color-scheme: dark) {
        button {
            color: white;
            background-color: #3a3a3a;
            border-color: #9a9a9a;
        }
    }
</style>
