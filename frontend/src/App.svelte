<script>
  import Router from 'svelte-spa-router'
  import Home from "./routes/Home.svelte"
  import Detail from "./routes/Detail.svelte"
  import Record from "./routes/Record.svelte"   // 추가

  const routes = {
    '/': Home,
    '/detail/:question_id': Detail,
    '/record': Record,    // 추가
  }
</script>

<Router {routes}/>