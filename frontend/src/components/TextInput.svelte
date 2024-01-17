<script>
    export let name;
</script>

<input on:change
    type="text"
    name={name}
    {...$$restProps} />

<style>
    input[type=text] {
    }
</style>
