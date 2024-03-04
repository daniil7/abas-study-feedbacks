<script>
    import AspectPicker from './incs/AspectsPicker.svelte';
    import FeedbacksTextsInput from './incs/FeedbacksTextsInput.svelte';
    import SubmitButton from './components/SubmitButton.svelte';

    let aspects = [];
    let texts = [];
    let result = {};
    let status = 'waiting';

    let api = new window.API();

    async function retriveResult() {
        result = []
        status = 'in-progress';
        result = await api.retriveResult(texts, aspects.length > 0 ? aspects : null);
        status = 'waiting';
    }
</script>

<main>
    <div>
        <h1 style="text-align: center;"> Course Aspects Analyser </h1>
        <hr />
        <AspectPicker bind:aspects={aspects} style="margin-bottom: 1rem;" />
        <FeedbacksTextsInput bind:texts={texts} />
        <SubmitButton on:click={retriveResult} text="Submit" />
        <div>
            {#if status == 'in-progress'}
            <p>Request in progress...</p>
            {:else}
            <ul>
                {#each Object.entries(result) as [aspect, info], index (aspect) }
                <li>{aspect} = {info.score}</li>
                {:else}
                <p>No results.</p>
                {/each}
            </ul>
            {/if}
        </div>
    </div>
</main>
