<script>
    import SubmitButton from '../components/SubmitButton.svelte'
    import CancelButton from '../components/CancelButton.svelte'
    import TextInput from '../components/TextInput.svelte'

    export let aspects = [];

    function addAspect(e)
    {
        const formData = new FormData(e.target);
        let data = {};
        formData.forEach((value, key) => data[key] = value);
        let aspect = data['aspect'];
        if (aspect != "" && !aspects.includes(aspect)) {
            aspects = [...aspects, aspect];
        }
        document.getElementById('aspect_input').value = "";
    }

    function removeAspect(e)
    {
        aspects = aspects.filter((aspect) => aspect != e.target.innerText);
    }

    function cleanAspects(e)
    {
        aspects = [];
    }
</script>

<div {...$$restProps}>

    <div style="display: flex; gap: 1rem;">
        <form style="display: flex; gap: 1rem; flex-grow: 1;" on:submit|preventDefault={addAspect}>
            <TextInput id="aspect_input" name="aspect" style="flex-grow: 1;" />
            <SubmitButton text="Add aspect" />
        </form>
        <CancelButton on:click={cleanAspects} text="╳" />
    </div>

    {#each aspects as aspect (aspect)}
        <div on:click={removeAspect} class="aspect">{aspect}</div>
    {/each}

</div>

<style>
    .aspect {
        display: inline-block;
        border-radius: 0.25rem;
        padding: 0.1rem 0.5rem 0.1rem 0.5rem;
        background-color: white;
        color: black;
        border: 1px solid black;
        cursor: pointer;
        margin: .25rem .25rem .25rem 0;
        user-select: none;
    }
</style>
