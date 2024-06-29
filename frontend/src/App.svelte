<script>
    import AspectPicker from './incs/AspectsPicker.svelte';
    import FeedbacksTextsInput from './incs/FeedbacksTextsInput.svelte';
    import SubmitButton from './components/SubmitButton.svelte';

    let api = new window.API();

    let aspects = []
    
    let texts = [];
    let result = {};
    let status = 'waiting';

    async function retriveResult() {
        result = []
        status = 'in-progress';
        result = await api.retriveResult(
            texts.slice(0, texts.length-1).map(t => t.text),
            aspects.length > 0 ? aspects : []
        );
        status = 'waiting';
    }

    (async () => {
        aspects = await api.retriveAspects();
    })();
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
                <li>{aspect} = {info}</li>
                {:else}
                <p>No results.</p>
                {/each}
            </ul>
            {/if}
        </div>
    </div>
</main>
