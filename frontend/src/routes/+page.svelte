
<script>
    import AspectPicker from '../incs/AspectsPicker.svelte';
    import FeedbacksTextsInput from '../incs/FeedbacksTextsInput.svelte';

    let aspects = [];
    let texts = [];
</script>

<div>
    <h1 style="text-align: center;"> Course Aspects Analyser </h1>
    <hr />
    <AspectPicker bind:aspects={aspects} style="margin-bottom: 1rem;" />
    <FeedbacksTextsInput bind:texts={texts} />
</div>
