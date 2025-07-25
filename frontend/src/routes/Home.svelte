<script>
    import fastapi from "../lib/api"
    import { link } from 'svelte-spa-router'

    let question_list = []

    fastapi('get', '/api/question/list', {}, (json) => {
        question_list = json
    })
</script>

<div class="container my-3">
    <!-- 버튼 -->
    <div class="mb-3">
        <a href="#/record" class="btn btn-primary">🎙️ 음성 녹음하러 가기</a>
    </div>
    
    <table class="table">
        <thead>
        <tr class="table-dark">
            <th>번호</th>
            <th>제목</th>
            <th>작성일시</th>
        </tr>
        </thead>
        <tbody>
        {#each question_list as question, i}
        <tr>
            <td>{i+1}</td>
            <td>
                <a use:link href="/detail/{question.id}">{question.subject}</a>
            </td>
            <td>{question.create_date}</td>
        </tr>
        {/each}
        </tbody>
    </table>
</div>