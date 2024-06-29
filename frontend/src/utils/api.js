export default class API
{
    constructor()
    {
        this.url = import.meta.env.VITE_API_URL;
        if (typeof this.url === 'undefined') {
            this.url = '/api';
        }
    }

    async retriveResult(texts, aspects = [])
    {
        return await (await fetch(`${this.url}/`, {
          method: "POST",
            body: JSON.stringify({
                texts: texts,
                aspect_labels: aspects
            }),
          headers: {
              "Content-type": "application/json; charset=UTF-8"
          }
        })).json();
    }

    async retriveAspects()
    {
        return await (await fetch(`${this.url}/aspects`, {
          method: "GET",
          headers: {
          }
        })).json();
    }
}
