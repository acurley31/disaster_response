import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'message'
})
export class MessagePipe implements PipeTransform {

    transform(value: any, ...args: any[]): any {
        let items = []
        for (var key in value) {
            if (key !== "id" && key !== "message") {
                const label = key.split("_").join(" ")
                const stat = value[key]
                items.push({ label: label, value: stat })
            }
        }

        console.log(items)
        
        return items;
    }

}
