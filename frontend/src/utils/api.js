export default class API
{
    constructor()
    {
        this.url = import.meta.env.VITE_API_URL;
        if (typeof this.url == 'undefined') {
            this.url = '/api';
        }
    }
    async retriveResult(texts, aspects = null)
    {
        return await (await fetch(this.url + "/", {
          method: "POST",
            body: JSON.stringify({
                text: texts.reduce(
                    (text, part) => text + part.text + ' ',
                    ''),
                aspect_labels: aspects
            }),
          headers: {
              "Content-type": "application/json; charset=UTF-8"
          }
        })).json();
    }
}
