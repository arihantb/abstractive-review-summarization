Chart.Chart.pluginService.register({
    beforeDraw: function(chart) {
        if (chart.config.centerText.display !== null &&
            typeof chart.config.centerText.display !== 'undefined' &&
            chart.config.centerText.display) {
            drawTotals(chart);
        }
    },
});

function drawTotals(chart) {

    var width = chart.chart.width,
    height = chart.chart.height,
    ctx = chart.chart.ctx;

    ctx.restore();
    var fontSize = (height / 114).toFixed(2);
    ctx.font = fontSize + "em 'Poppins', sans-serif";
    ctx.textBaseline = "middle";

    var text = chart.config.centerText.text,
    textX = Math.round((width - ctx.measureText(text).width) / 2),
    textY = height / 1.9;

    ctx.fillText(text, textX, textY);
    ctx.save();
}
