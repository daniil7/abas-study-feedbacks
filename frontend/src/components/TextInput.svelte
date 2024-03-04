<script>
    export let name;
</script>

<input on:change on:input
    type="text"
    name={name}
    {...$$restProps} />

<style>
    input[type=text] {
    }
</style>
