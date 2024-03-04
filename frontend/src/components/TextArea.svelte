
<script>
    export let value = "";
</script>

<textarea on:change on:input
    {...$$restProps}>{value}</textarea>

<style>
    textarea {
    }
</style>
