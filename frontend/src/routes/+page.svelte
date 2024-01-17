
<script>
    import AspectPicker from '../incs/AspectsPicker.svelte';
    let aspects = [];
</script>

<div>
    <h1 style="text-align: center;"> Course Aspects Analyser </h1>
    <hr />
    <AspectPicker bind:aspects={aspects} style="margin-bottom: 1rem;" />
</div>
