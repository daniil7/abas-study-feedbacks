<script>
    import { onMount } from 'svelte';
    import TextArea from '../components/TextArea.svelte'

    export let texts = [];

	onMount(() => {
        texts = [{id: 0, text: ""}];
		return () => {
			cancelAnimationFrame(frame);
	    };
    });

    function TextUpdated(e)
    {
        let user_input_id = e.target.getAttribute('feedback-text-id');
        let user_input_text = e.target.value;
        let end_id = texts[texts.length - 1].id

        if (user_input_text == "") {
            if (user_input_id != end_id) {
                texts = texts.filter((text) => text.id != user_input_id);
            }
        } else {
            let index = texts.findIndex((text) => text.id == user_input_id);
            texts[index].text = user_input_text;

            if (user_input_id == end_id)
            {
                texts = [...texts, {id: end_id + 1, text: ""}];
            }
        }
        console.log(texts);
    }

</script>

<div {...$$restProps}>

    {#each texts as text (text.id)}
        <TextArea style="width: 100%; margin-bottom: 1rem"
            feedback-text-id={text.id}
            on:change={TextUpdated}
            value={text.value} />
    {/each}

</div>
