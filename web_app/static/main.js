async function fetchData() {

    let g = await fetch('/gesture').then(r => r.json());
    document.getElementById("gesture").innerText = g.gesture;

    let s = await fetch('/speech').then(r => r.json());
    document.getElementById("speech").innerText = s.text;

    let sum = await fetch('/summary').then(r => r.json());
    document.getElementById("summary").innerText = sum.summary;

    let k = await fetch('/keywords').then(r => r.json());
    document.getElementById("keywords").innerHTML =
        k.keywords.map(w => `<li>${w}</li>`).join("");

    let a = await fetch('/attendance').then(r => r.json());
    document.getElementById("attendance").innerHTML =
        a.map(row =>
            `<tr>
                <td>${row[0]}</td>
                <td>${row[1]}</td>
                <td>${row[2]}</td>
                <td>${row[3]}</td>
            </tr>`
        ).join("");
}

setInterval(fetchData, 1000);
fetchData();