async function treinarEPrever() {

            const textoStatus = document.getElementById("status");
            const textoResultado = document.getElementById("resultado");

            const minutosDigitados = Number(document.getElementById("minutos").value);

    if (isNaN(minutosDigitados) || minutosDigitados <= 0) {
    textoResultado.innerText = "Digite um valor válido de minutos!";
    textoStatus.innerText = "Status: erro nos dados";
    return;
}

            textoStatus.innerText = "Status: Treinando a IA...";

            // =========================
            // 1. CRIAR O MODELO
            // =========================
            const modelo = tf.sequential();
            modelo.add(tf.layers.dense({
                units: 1,
                inputShape: [1]
            }));

            // =========================
            // 2. COMPILAR
            // =========================
            modelo.compile({
                loss: 'meanSquaredError',
                optimizer: 'sgd'
            });

            // =========================
            // 3. DADOS DE TREINO
            // minutos → calorias
            // =========================
            const dadosEntrada = tf.tensor2d([10, 20, 30, 45, 60], [5, 1]);
            const dadosSaida = tf.tensor2d([60, 120, 180, 280, 360], [5, 1]);

            // =========================
            // 4. TREINAMENTO
            // =========================
            await modelo.fit(dadosEntrada, dadosSaida, {
                epochs: 200
            });

            textoStatus.innerText = "Status: IA treinada!";

            // =========================
            // 5. PREVISÃO
            // =========================
            const previsao = modelo.predict(
                tf.tensor2d([minutosDigitados], [1, 1])
            );

            const valorPrevisto = previsao.dataSync()[0];

            textoResultado.innerText =
                "Calorias estimadas: " + valorPrevisto.toFixed(2) + " kcal";
        }