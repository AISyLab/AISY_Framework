class Layout {
    constructor(x_axis_title, y_axis_title) {
        this.x_axis_title = x_axis_title;
        this.y_axis_title = y_axis_title;
    }

    plotly_layout() {
        return {
            annotations: [],
            font: {
                size: 12,
                family: 'Calibri',
                color: '#fff'
            },
            paper_bgcolor: '#263238',
            plot_bgcolor: '#263238',
            xaxis: {
                ticks: '',
                side: 'bottom',
                title: this.x_axis_title,
                tickcolor: '#fff',
                gridcolor: '#616161',
                color: 'fff'
            },
            yaxis: {
                ticks: '',
                ticksuffix: ' ',
                autosize: false,
                title: this.y_axis_title,
                tickcolor: '#fff',
                gridcolor: '#616161',
                color: 'fff'
            }
        };
    }
}