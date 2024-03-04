import './app.css'
import App from './App.svelte'

import API from './utils/api.js'
window.API = API

const app = new App({
  target: document.getElementById('app'),
})

export default app
